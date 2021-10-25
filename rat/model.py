import blackjax
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
from .fit import Fit


class Model:
    lpdf: Callable[[numpy.array], float]
    size: int
    parameter_variables: List[variables.Param]
    parameter_names: List[str]
    parameter_offsets: List[int]
    parameter_sizes: List[Union[None, int]]

    def __init__(self, data_df: pandas.DataFrame, parsed_lines: List[ops.Expr]):
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
            data_numpy_variables[name] = data.to_numpy()

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
                    parameter, constraints_jacobian_adjustment = constraints.lower(
                        parameter, lower
                    )
                elif lower == float("inf") and upper < float("inf"):
                    parameter, constraints_jacobian_adjustment = constraints.upper(
                        parameter, upper
                    )
                elif lower > float("inf") and upper < float("inf"):
                    parameter, constraints_jacobian_adjustment = constraints.finite(
                        parameter, lower, upper
                    )

                if size is not None and size != variable.padded_size():
                    parameter = jax.numpy.pad(
                        parameter, (0, variable.padded_size() - size)
                    )

                if include_jacobian:
                    total += constraints_jacobian_adjustment
                parameter_numpy_variables[name] = parameter

            for line_function in line_functions:
                data_arguments = [
                    data_numpy_variables[name]
                    for name in line_function.data_variable_names
                ]
                parameter_arguments = [
                    parameter_numpy_variables[name]
                    for name in line_function.parameter_variable_names
                ]
                total += line_function(*data_arguments, *parameter_arguments)

            return total

        self.parameter_variables = parameter_variables
        self.lpdf = jax.jit(functools.partial(lpdf, True))
        self.lpdf_no_jac = jax.jit(functools.partial(lpdf, False))
        self.size = unconstrained_parameter_size

    def optimize(self) -> Fit:
        params = numpy.random.rand(self.size)

        nlpdf = lambda x: -self.lpdf_no_jac(x.astype(numpy.float32))
        grad = jax.jit(jax.grad(nlpdf))
        grad_double = lambda x: numpy.array(grad(x.astype(numpy.float32))).astype(
            numpy.float64
        )

        results = scipy.optimize.minimize(
            nlpdf, params, jac=grad_double, method="L-BFGS-B", tol=1e-9
        )

        if not results.success:
            raise Exception(f"Optimization failed: {results.message}")

        draw_dfs: Dict[str, pandas.DataFrame] = {}
        for name, offset, size in zip(
            self.parameter_names, self.parameter_offsets, self.parameter_sizes
        ):
            if size is not None:
                df = self.parameter_variables[name].index.base_df.copy()
                df["value"] = results.x[offset : offset + size]
            else:
                df = pandas.DataFrame({"value": [results.x[offset]]})

            variable = self.parameter_variables[name]
            lower = variable.lower
            upper = variable.upper
            value = df["value"].to_numpy()
            if lower > float("-inf") and upper == float("inf"):
                value, _ = constraints.lower(value, lower)
            elif lower == float("inf") and upper < float("inf"):
                value, _ = constraints.upper(value, upper)
            elif lower > float("inf") and upper < float("inf"):
                value, _ = constraints.finite(value, lower, upper)
            df["value"] = value

            draw_dfs[name] = df

        return Fit(draw_dfs)

    def sample(self, num_steps=200, step_size=1e-3) -> Fit:
        params = numpy.random.rand(self.size)

        # Build the kernel
        inverse_mass_matrix = jax.numpy.exp(jax.numpy.zeros(self.size))
        kernel = blackjax.nuts.kernel(self.lpdf, step_size, inverse_mass_matrix)
        kernel = jax.jit(kernel)  # try without to see the speedup

        # Initialize the state
        initial_position = params
        state = blackjax.nuts.new_state(initial_position, self.lpdf)

        states: List[blackjax.inference.base.HMCState] = [state]

        # Iterate
        key = jax.random.PRNGKey(0)
        for draw in range(1, num_steps):
            key, subkey = jax.random.split(key)
            state, info = kernel(key, state)
            states.append(state)

        draw_dfs: Dict[str, pandas.DataFrame] = {}
        draw_series = list(range(num_steps))
        for name, offset, size in zip(
            self.parameter_names, self.parameter_offsets, self.parameter_sizes
        ):
            if size is not None:
                dfs = []
                for draw, state in enumerate(states):
                    df = self.parameter_variables[name].index.base_df.copy()
                    df[name] = state.position[offset : offset + size]
                    df["draw"] = draw
                    dfs.append(df)
                draw_dfs[name] = pandas.concat(dfs, ignore_index=True)
            else:
                series = [state.position[offset] for state in states]
                draw_dfs[name] = pandas.DataFrame({name: series, "draw": draw_series})

        return Fit(draw_dfs)
