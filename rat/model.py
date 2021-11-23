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
from .scanner import scanner
from .parser import Parser
from .fit import Fit


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
            assigned_parameter_variables,
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

        # identify which values are needed for each assigned_param
        assigned_param_dependencies = {}
        for name, param in assigned_parameter_variables.items():
            dependant_data = set()
            dependant_param = set()
            dependant_assigned_param = set()
            for expr in ops.search_tree(param.rhs, ops.Data, ops.Param):
                if isinstance(expr, ops.Data):
                    dependant_data.add(expr.get_key())
                elif isinstance(expr, ops.Param):
                    if expr.get_key() in assigned_parameter_variables:
                        dependant_assigned_param.add(expr.get_key())
                    else:
                        dependant_param.add(expr.get_key())

            assigned_param_dependencies[name] = {"param": tuple(dependant_param), "data": tuple(dependant_data), "assigned_param": tuple(dependant_assigned_param)}

        # This is the likelihood function we'll expose!
        def lpdf(include_jacobian, unconstrained_parameter_vector):
            parameter_numpy_variables = {}
            assigned_parameter_numpy_variables = {}
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

            # evaluate assigned_parameters
            for name, param in assigned_parameter_variables.items():
                size = param.size()
                if size:
                    assigned_param_array = jax.numpy.zeros(size)
                else:
                    assigned_param_array = jax.numpy.zeros(1)

                local_vars = {}  # this will hold the rhs variables for evaluation
                # if assigned_param_dependencies[name]["param"]:
                #     print(assigned_param_array.shape)
                #     print(param.ops_param.index.code())

                for val in assigned_param_dependencies[name]["param"]:
                    par = parameter_variables[val]
                    local_vars[par.code()] = parameter_numpy_variables[val]

                for val in assigned_param_dependencies[name]["data"]:
                    par = data_variables[val]
                    local_vars[par.code()] = data_variables[val]

                for val in assigned_param_dependencies[name]["assigned_param"]:
                    par = data_variables[val]
                    local_vars[par.code()] = assigned_parameter_variables[val]

                # this is such a horrible method I'm not even joking
                # this assumes all variables share the same subscripts
                if param.index:
                    assigned_parameter_numpy_variables[name] = eval(param.rhs.code().replace(f"[{param.ops_param.index.code()}]", ""), globals(), local_vars)
                else:
                    assigned_parameter_numpy_variables[name] = eval(param.rhs.code(), globals(), local_vars)


            for line_function in line_functions:
                data_arguments = [data_numpy_variables[name] for name in line_function.data_variable_names]
                parameter_arguments = [parameter_numpy_variables[name] for name in line_function.parameter_variable_names]
                assigned_parameter_arguments = [assigned_parameter_numpy_variables[name] for name in line_function.assigned_parameter_variables_names]
                total += line_function(*data_arguments, *parameter_arguments, *assigned_parameter_arguments)

            return total

        self.parameter_variables = parameter_variables
        self.lpdf = jax.jit(functools.partial(lpdf, True))
        self.lpdf_no_jac = jax.jit(functools.partial(lpdf, False))
        self.size = unconstrained_parameter_size

    def optimize(self, init=2) -> Fit:
        params = 2 * init * numpy.random.rand(self.size) - init

        nlpdf = lambda x: -self.lpdf_no_jac(x.astype(numpy.float32))
        grad = jax.jit(jax.grad(nlpdf))
        grad_double = lambda x: numpy.array(grad(x.astype(numpy.float32))).astype(numpy.float64)

        results = scipy.optimize.minimize(nlpdf, params, jac=grad_double, method="L-BFGS-B", tol=1e-7)

        if not results.success:
            raise Exception(f"Optimization failed: {results.message}")

        draw_dfs: Dict[str, pandas.DataFrame] = {}
        for name, offset, size in zip(self.parameter_names, self.parameter_offsets, self.parameter_sizes):
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
        for name, offset, size in zip(self.parameter_names, self.parameter_offsets, self.parameter_sizes):
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
