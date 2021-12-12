import blackjax
import blackjax.stan_warmup
import blackjax.nuts
import functools
import jax
import jax.scipy
import jax.scipy.optimize
import jax.numpy
import logging
import numpy
import pandas
import scipy.optimize
from typing import Callable, List, Dict, Union

from . import compiler
from . import ops
from . import variables
from . import constraints
from .scanner import scanner
from .parser import Parser
from . import fit


class Model:
    lpdf: Callable[[numpy.array], float]
    size: int
    parameter_variables: List[variables.Param]
    parameter_names: List[str]
    parameter_offsets: List[int]
    parameter_sizes: List[Union[None, int]]

    def constrain(self, unconstrained_parameter_vector, pad=True):
        constrained_variables = {}
        total = 0.0
        for name, offset, size in zip(
            self.parameter_names,
            self.parameter_offsets,
            self.parameter_sizes,
        ):
            if size is not None:
                parameter = unconstrained_parameter_vector[offset : offset + size]
            else:
                parameter = unconstrained_parameter_vector[offset]

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

    def __init__(
        self,
        data_df: pandas.DataFrame,
        parsed_lines: List[ops.Expr] = None,
        model_string: str = None,
    ):
        data_names = data_df.columns
        if model_string is not None:
            if parsed_lines is not None:
                raise Exception("Only one of model_string and parsed_lines can be non-None")

            parsed_lines = []
            for line in model_string.splitlines():
                line = line.lstrip().rstrip()
                if not line:
                    continue
                parsed_line = Parser(scanner(line), data_names, line).statement()
                parsed_lines.append(parsed_line)
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
        device_variables = {}
        for name, data in data_variables.items():
            device_variables[name] = jax.device_put(data.to_numpy())

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

        # This is the likelihood function we'll expose!
        def lpdf(include_jacobian, unconstrained_parameter_vector):
            total = 0.0
            jacobian_adjustment, constrained_variables = self.constrain(unconstrained_parameter_vector)
            device_variables.update(constrained_variables)
            if include_jacobian:
                total = jacobian_adjustment

            for line_function in line_functions:
                data_arguments = [device_variables[name] for name in line_function.data_variable_names]
                parameter_arguments = [device_variables[name] for name in line_function.parameter_variable_names]
                assigned_parameter_arguments = [device_variables[name] for name in line_function.assigned_parameter_variables_names]
                if isinstance(line_function, compiler.AssignLineFunction):
                    device_variables[line_function.name] = line_function(
                        *data_arguments, *parameter_arguments, *assigned_parameter_arguments
                    )
                else:
                    total += line_function(*data_arguments, *parameter_arguments, *assigned_parameter_arguments)

            return total

        self.parameter_variables = parameter_variables
        self.lpdf = jax.jit(functools.partial(lpdf, True))
        self.lpdf_no_jac = jax.jit(functools.partial(lpdf, False))
        self.size = unconstrained_parameter_size

    def prepare_draws_and_dfs(self, unconstrained_draws):
        jacobian_adjustment, device_constrained_draws = self.constrain(unconstrained_draws, pad=False)
        # Copy back to numpy arrays
        constrained_draws = {key: numpy.array(value) for key, value in device_constrained_draws.items()}
        base_dfs = {}
        for name in constrained_draws:
            if self.parameter_variables[name].index is not None:
                base_dfs[name] = self.parameter_variables[name].index.base_df.copy()
            else:
                base_dfs[name] = pandas.DataFrame()
        return constrained_draws, base_dfs

    def optimize(self, init=2, chains=4, retries=5, tolerance=1e-2):
        def nlpdf(x):
            return -self.lpdf_no_jac(x.astype(numpy.float32))

        grad = jax.jit(jax.grad(nlpdf))

        def grad_double(x):
            grad_device_array = grad(x)
            return numpy.array(grad_device_array).astype(numpy.float64)

        unconstrained_draws = numpy.zeros((self.size, 1, chains))
        for chain in range(chains):
            for retry in range(retries):
                params = 2 * init * numpy.random.uniform(size=self.size) - init

                solution = scipy.optimize.minimize(nlpdf, params, jac=grad_double, method="L-BFGS-B", tol=1e-9)

                if solution.success:
                    unconstrained_draws[:, 0, chain] = solution.x
                    break
            else:
                raise Exception(f"Optimization failed on chain {chain} with message: {solution.message}")

        constrained_draws, base_dfs = self.prepare_draws_and_dfs(unconstrained_draws)
        return fit.OptimizationFit(constrained_draws, base_dfs, tolerance=tolerance)

    def sample(self, num_draws=200, num_warmup=200, chains=4, init=2, step_size=1e-2):
        # Currently only doing warmup on one chain
        initial_position = 2 * init * numpy.random.uniform(size=(self.size)) - init

        def kernel_generator(step_size, inverse_mass_matrix):
            return blackjax.nuts.kernel(self.lpdf, step_size, inverse_mass_matrix)

        # inverse_mass_matrix = jax.numpy.exp(jax.numpy.zeros(self.size))
        # kernel = blackjax.nuts.kernel(self.lpdf, step_size, inverse_mass_matrix)
        # kernel = jax.jit(kernel)

        # Initialize the state
        initial_state = blackjax.nuts.new_state(initial_position, self.lpdf)
        # positions = jax.numpy.array(chains * [initial_position])
        # initial_states = jax.vmap(blackjax.nuts.new_state, in_axes = (0, None))(positions, self.lpdf)

        # Do one-chain warmup
        key = jax.random.PRNGKey(0)
        warmup_key, key = jax.random.split(key)
        state, (step_size, inverse_mass_matrix), info = blackjax.stan_warmup.run(
            warmup_key,
            kernel_generator,
            initial_state,
            num_warmup,
        )

        kernel = jax.jit(kernel_generator(step_size, inverse_mass_matrix))

        positions = jax.numpy.array(chains * [state.position])
        states = jax.vmap(blackjax.nuts.new_state, in_axes=(0, None))(positions, self.lpdf)

        # Sample chains
        def one_step(chain_states, rng_key):
            keys = jax.random.split(rng_key, chains)
            new_chain_states, _ = jax.vmap(kernel)(keys, chain_states)
            return new_chain_states, new_chain_states

        keys = jax.random.split(key, num_draws)
        _, states = jax.lax.scan(one_step, states, keys)

        # Reorder as (param, draws, chains) from (draws, chains, param)
        unconstrained_draws = numpy.moveaxis(states.position, (0, 1, 2), (1, 2, 0))

        constrained_draws, base_dfs = self.prepare_draws_and_dfs(unconstrained_draws)

        return fit.SampleFit(constrained_draws, base_dfs)
