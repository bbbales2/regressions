import blackjax
import blackjax.stan_warmup
import blackjax.nuts
import functools
import jax
import jax.scipy
import jax.scipy.optimize
import jax.numpy
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
                parsed_line = Parser(scanner(line), data_names).statement()
                parsed_lines.append(parsed_line)
        else:
            if parsed_lines is None:
                raise Exception("At least one of model_string or parsed_lines must be non-None")

        (
            data_variables,
            parameter_variables,
            index_variables,
            line_functions,
        ) = compiler.compile(data_df, parsed_lines)

        self.parameter_names = []
        self.parameter_offsets = []
        self.parameter_sizes = []

        # Copy data to jax device
        data_numpy_variables = {}
        for name, data in data_variables.items():
            data_numpy_variables[name] = jax.device_put(data.to_numpy())

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
            parameter_numpy_variables = {}
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

                if size is not None and size != variable.padded_size():
                    parameter = jax.numpy.pad(parameter, (0, variable.padded_size() - size))

                if include_jacobian:
                    total += constraints_jacobian_adjustment
                parameter_numpy_variables[name] = parameter

            for line_function in line_functions:
                data_arguments = [data_numpy_variables[name] for name in line_function.data_variable_names]
                parameter_arguments = [parameter_numpy_variables[name] for name in line_function.parameter_variable_names]
                total += line_function(*data_arguments, *parameter_arguments)

            return total

        self.parameter_variables = parameter_variables
        self.lpdf = jax.jit(functools.partial(lpdf, True))
        self.lpdf_no_jac = jax.jit(functools.partial(lpdf, False))
        self.size = unconstrained_parameter_size

    def optimize(self, init=2, chains=4, retries=5, tolerance = 1e-2):
        def nlpdf(x):
            return -self.lpdf_no_jac(x.astype(numpy.float32))

        grad = jax.jit(jax.grad(nlpdf))

        def grad_double(x):
            grad_device_array = grad(x)
            return numpy.array(grad_device_array).astype(numpy.float64)

        unconstrained_draws = numpy.zeros((chains, 1, self.size))
        for chain in range(chains):
            params = 2 * init * numpy.random.uniform(size = self.size) - init

            for retry in range(retries):
                solution = scipy.optimize.minimize(
                    nlpdf, params, jac=grad_double, method="L-BFGS-B", tol=1e-7
                )

                if solution.success:
                    unconstrained_draws[chain, 0] = solution.x
                    break
            else:
                raise Exception(
                    f"Optimization failed on chain {chain} with message: {solution.message}"
                )

        return fit.OptimizationFit(self, unconstrained_draws, tolerance = tolerance)

    def sample(self, num_draws=200, num_warmup=200, chains=4, init=2, step_size=1e-2):
        initial_positions = 2 * init * numpy.random.uniform(size = (chains, self.size)) - init

        # Build the kernel
        def kernel_generator(step_size, inverse_mass_matrix):
            return blackjax.nuts.kernel(self.lpdf, step_size, inverse_mass_matrix)

        inverse_mass_matrix = jax.numpy.exp(jax.numpy.zeros(self.size))
        #kernel = blackjax.nuts.kernel(self.lpdf, step_size, inverse_mass_matrix)
        #kernel = jax.jit(kernel)

        # Initialize the state
        states: List[blackjax.inference.base.HMCState] = []
        for initial_position in initial_positions:
            states.append(blackjax.nuts.new_state(initial_position, self.lpdf))

        # Do warmup for each chain
        key = jax.random.PRNGKey(0)
        # states = []
        kernels = []
        for chain in range(chains):
            key, subkey = jax.random.split(key)
            state, (step_size, inverse_mass_matrix), info = blackjax.stan_warmup.run(
                key,
                kernel_generator,
                initial_states[chain],
                num_warmup,
            )
            states.append(state)
            kernels.append(jax.jit(kernel_generator(step_size, inverse_mass_matrix)))

        # Initialize storage for draws
        unconstrained_draws = numpy.zeros((chains, num_draws, self.size))

        # Iterate
        for chain in range(0, chains):
            unconstrained_draws[chain, 0] = states[chain].position
            for draw in range(1, num_draws):
                key, subkey = jax.random.split(key)
                states[chain], info = kernels[chain](key, states[chain])
                unconstrained_draws[chain, draw] = states[chain].position
        
        return fit.SampleFit(self, unconstrained_draws)
