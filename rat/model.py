import blackjax
import blackjax.stan_warmup
import blackjax.nuts
import functools
import itertools
import jax
import jax.experimental.host_callback
import jax.scipy
import jax.scipy.optimize
import jax.numpy
import logging
import numpy
import pandas
import scipy.optimize
from typing import Callable, List, Dict, Union
from tqdm import tqdm

from . import compiler
from . import ops
from . import variables
from . import constraints
from .scanner import Scanner
from .parser import Parser
from . import fit


class Model:
    log_density_jax: Callable[[numpy.array], float]
    log_density_jax_no_jac: Callable[[numpy.array], float]
    size: int
    parameter_variables: List[variables.Param]
    parameter_names: List[str]
    parameter_offsets: List[int]
    parameter_sizes: List[Union[None, int]]

    def _constrain(self, unconstrained_parameter_vector, pad=True):
        constrained_variables = {}
        total = 0.0
        for name, offset, size in zip(
            self.parameter_names,
            self.parameter_offsets,
            self.parameter_sizes,
        ):
            if size is not None:
                parameter = unconstrained_parameter_vector[..., offset : offset + size]
            else:
                parameter = unconstrained_parameter_vector[..., offset]

            variable = self.parameter_variables[name]
            lower = variable.lower
            upper = variable.upper
            constraints_jacobian_adjustment = 0.0

            if lower > float("-inf") and upper == float("inf"):
                parameter, constraints_jacobian_adjustment = constraints.lower(parameter, lower)
            elif lower == float("inf") and upper < float("inf"):
                parameter, constraints_jacobian_adjustment = constraints.upper(parameter, upper)
            elif lower > float("inf") and upper < float("inf"):
                parameter, constraints_jacobian_adjustment = constraints.finite(parameter, lower, upper)

            if pad and size is not None and size != variable.padded_size():
                parameter = jax.numpy.pad(parameter, (0, variable.padded_size() - size))

            total += jax.numpy.sum(constraints_jacobian_adjustment)
            constrained_variables[name] = parameter

        return total, constrained_variables

    def _evaluate_program(self, **device_variables):
        total = 0.0
        for line_function in self.line_functions:
            data_arguments = [device_variables[name] for name in line_function.data_variable_names]
            parameter_arguments = [device_variables[name] for name in line_function.parameter_variable_names]
            assigned_parameter_arguments = [device_variables[name] for name in line_function.assigned_parameter_variables_names]
            if isinstance(line_function, compiler.AssignLineFunction):
                device_variables[line_function.name] = line_function(*data_arguments, *parameter_arguments, *assigned_parameter_arguments)
            else:
                total += line_function(*data_arguments, *parameter_arguments, *assigned_parameter_arguments)

        return total, device_variables

    def _log_density(self, include_jacobian, unconstrained_parameter_vector):
        jacobian_adjustment, constrained_variables = self._constrain(unconstrained_parameter_vector)
        constrained_variables.update(self.device_data_variables)

        target_program, constrained_variables = self._evaluate_program(**constrained_variables)
        return target_program + (jacobian_adjustment if include_jacobian else 0.0)

    def _prepare_draws_and_dfs(self, device_unconstrained_draws):
        unconstrained_draws = numpy.array(device_unconstrained_draws)
        num_draws = unconstrained_draws.shape[0]
        num_chains = unconstrained_draws.shape[1]
        variables_to_extract = set(itertools.chain(self.parameter_variables.keys(), self.assigned_parameter_variables.keys()))
        constrained_draws = {}
        evaluate_program_with_data = jax.jit(functools.partial(self._evaluate_program, **self.device_data_variables), backend="cpu")
        for draw in range(num_draws):
            for chain in range(num_chains):
                jacobian_adjustment, constrained_variables = self._constrain(unconstrained_draws[draw, chain], pad=False)
                device_constrained_variables = {k: jax.numpy.array(v) for k, v in constrained_variables.items()}
                target, device_constrained_variables = evaluate_program_with_data(**device_constrained_variables)

                for name, device_constrained_variable in device_constrained_variables.items():
                    if name in variables_to_extract:
                        if name not in constrained_draws:
                            constrained_draws[name] = numpy.zeros((num_draws, num_chains) + device_constrained_variable.shape)
                        constrained_draws[name][draw, chain] = numpy.array(device_constrained_variable)

        ## This is probably faster than above but uses more device memory
        # jacobian_adjustment, device_constrained_variables = self._constrain(unconstrained_draws, pad=False)
        # device_constrained_variables = {k : jax.numpy.array(v) for k, v in device_constrained_variables.items()}
        # target, device_constrained_variables = jax.vmap(jax.vmap(evaluate_program_with_data))(**device_constrained_variables)

        # Copy back to numpy arrays
        base_dfs = {}
        for name, variable in itertools.chain(self.parameter_variables.items(), self.assigned_parameter_variables.items()):
            if variable.index is not None:
                base_dfs[name] = variable.index.base_df.copy()
            else:
                base_dfs[name] = pandas.DataFrame()
        return constrained_draws, base_dfs

    def __init__(
        self,
        data_df: pandas.DataFrame,
        model_string: str = None,
        parsed_lines: List[ops.Expr] = None,
    ):
        """
        Create a model from a dataframe (`data_df`) and a model (specified as a string, `model_string`)

        The parsed_lines argument is for creating a model from an intermediate representation -- likely
        deprecated soon.
        """
        data_names = data_df.columns
        if model_string is not None:
            if parsed_lines is not None:
                raise Exception("Only one of model_string and parsed_lines can be non-None")

            parsed_lines = []
            scanned_lines = Scanner(model_string).scan()
            for scanned_line in scanned_lines:
                parsed_lines.append(Parser(scanned_line, data_names, model_string).statement())
        else:
            if parsed_lines is None:
                raise Exception("At least one of model_string or parsed_lines must be non-None")

        (
            data_variables,
            parameter_variables,
            assigned_parameter_variables,
            index_variables,
            line_functions,
        ) = compiler.compile(data_df, parsed_lines)

        self.parameter_names = []
        self.parameter_offsets = []
        self.parameter_sizes = []

        # Copy data to jax device
        device_data_variables = {}
        for name, data in data_variables.items():
            device_data_variables[name] = jax.device_put(data.to_numpy())

        # Get parameter dimensions so can build individual arguments from
        # unconstrained vector
        unconstrained_parameter_size = 0
        for name, parameter in parameter_variables.items():
            self.parameter_names.append(name)
            self.parameter_offsets.append(unconstrained_parameter_size)
            parameter_size = parameter.size()
            self.parameter_sizes.append(parameter_size)

            if parameter_size is not None:
                unconstrained_parameter_size += parameter_size
            else:
                unconstrained_parameter_size += 1

        self.line_functions = line_functions
        self.parameter_variables = parameter_variables
        self.assigned_parameter_variables = assigned_parameter_variables
        self.device_data_variables = device_data_variables
        self.log_density_jax = jax.jit(functools.partial(self._log_density, True))
        self.log_density_jax_no_jac = jax.jit(functools.partial(self._log_density, False))
        self.size = unconstrained_parameter_size

    def optimize(self, init=2, chains=4, retries=5, tolerance=1e-2):
        """
        Maximize the log density. `chains` difference optimizations are initialized.

        An error is thrown if the different solutions are not all within tolerance of the
        median solution for each parameter. If only one chain is used, the tolerance is
        ignored.

        If any optimization fails, retry up to `retries` number of times.

        Initialize parameters in unconstrained space uniformly [-2, 2].
        """

        def negative_log_density(x):
            return -self.log_density_jax_no_jac(x.astype(numpy.float32))

        grad = jax.jit(jax.grad(negative_log_density))

        def grad_double(x):
            grad_device_array = grad(x)
            return numpy.array(grad_device_array).astype(numpy.float64)

        unconstrained_draws = numpy.zeros((1, chains, self.size))
        for chain in range(chains):
            for retry in range(retries):
                params = 2 * init * numpy.random.uniform(size=self.size) - init

                solution = scipy.optimize.minimize(negative_log_density, params, jac=grad_double, method="L-BFGS-B", tol=1e-9)

                if solution.success:
                    unconstrained_draws[0, chain] = solution.x
                    break
            else:
                raise Exception(f"Optimization failed on chain {chain} with message: {solution.message}")

        constrained_draws, base_dfs = self._prepare_draws_and_dfs(unconstrained_draws)
        return fit.OptimizationFit._from_constrained_variables(constrained_draws, base_dfs, tolerance=tolerance)

    def sample(self, num_draws=200, num_warmup=200, chains=4, init=2, target_acceptance_rate=0.85):
        """
        Sample the target log density using NUTS.

        Sample using `chains` different chains with parameters initialized in unconstrained
        space [-2, 2]. Use `num_warmup` draws to warmup and collect `num_draws` draws in each
        chain after warmup.

        Regardless of the value of `chains`, only one chain is used for warmup.

        `target_acceptance_rate` is the target acceptance rate for adaptation. Should be less
        than one and greater than zero.
        """
        # Currently only doing warmup on one chain
        initial_position = 2 * init * numpy.random.uniform(size=(self.size)) - init

        assert target_acceptance_rate < 1.0 and target_acceptance_rate > 0.0

        step_size = 1.0

        def kernel_generator(step_size, inverse_mass_matrix):
            return blackjax.nuts.kernel(self.log_density_jax, step_size, inverse_mass_matrix)

        # inverse_mass_matrix = jax.numpy.exp(jax.numpy.zeros(self.size))
        # kernel = blackjax.nuts.kernel(self.lpdf, step_size, inverse_mass_matrix)
        # kernel = jax.jit(kernel)

        # Initialize the state
        initial_state = blackjax.nuts.new_state(initial_position, self.log_density_jax)
        # positions = jax.numpy.array(chains * [initial_position])
        # initial_states = jax.vmap(blackjax.nuts.new_state, in_axes = (0, None))(positions, self.lpdf)

        # Do one-chain warmup
        key = jax.random.PRNGKey(0)
        warmup_key, key = jax.random.split(key)

        ## The warmup here is copy-pasted and modified from blackjax source
        ## This is the blackjax warmup call we're replacing
        # state, (step_size, inverse_mass_matrix), info = blackjax.stan_warmup.run(
        #     warmup_key,
        #     kernel_generator,
        #     initial_state,
        #     num_warmup,
        # )

        init, update, final = blackjax.stan_warmup.stan_warmup(
            kernel_generator,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=target_acceptance_rate,
        )

        with tqdm(total=num_warmup, desc="Warming up") as progress_bar:

            def progress_callback(iteration, transforms):
                progress_bar.update()

            def one_step_warmup(carry_and_iteration, interval):
                (rng_key, state, warmup_state), iteration = carry_and_iteration
                stage, is_middle_window_end = interval
                jax.experimental.host_callback.id_tap(progress_callback, iteration)
                _, rng_key = jax.random.split(rng_key)
                state, warmup_state, info = update(rng_key, stage, is_middle_window_end, state, warmup_state)

                return (((rng_key, state, warmup_state), iteration + 1), (state, warmup_state, info))

            schedule = jax.numpy.array(blackjax.stan_warmup.stan_warmup_schedule(num_warmup))
            warmup_state = init(warmup_key, initial_state, step_size)
            last_state, warmup_chain = jax.lax.scan(one_step_warmup, ((warmup_key, initial_state, warmup_state), 0), schedule)
        (_, last_chain_state, last_warmup_state), i = last_state

        step_size, inverse_mass_matrix = final(last_warmup_state)
        state = last_chain_state
        info = warmup_chain

        kernel = jax.jit(kernel_generator(step_size, inverse_mass_matrix))

        positions = jax.numpy.array(chains * [state.position])
        states = jax.vmap(blackjax.nuts.new_state, in_axes=(0, None))(positions, self.log_density_jax)

        with tqdm(total=num_draws, desc="  Sampling") as progress_bar:

            def progress_callback(iteration, transforms):
                progress_bar.update()

            # Sample chains
            def one_step(chain_states_and_iteration, rng_key):
                chain_states, iteration = chain_states_and_iteration
                jax.experimental.host_callback.id_tap(progress_callback, iteration)
                keys = jax.random.split(rng_key, chains)
                new_chain_states, _ = jax.vmap(kernel)(keys, chain_states)
                return (new_chain_states, iteration + 1), new_chain_states

            keys = jax.random.split(key, num_draws)
            _, states = jax.lax.scan(one_step, (states, 0), keys)

        # Ordered as (draws, chains, param)
        unconstrained_draws = states.position

        constrained_draws, base_dfs = self._prepare_draws_and_dfs(unconstrained_draws)

        return fit.SampleFit._from_constrained_variables(constrained_draws, base_dfs)
