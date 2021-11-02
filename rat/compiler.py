import numpy
import pandas
import jax
import jax.numpy as jnp
import jax.scipy.stats
from typing import Iterable, List, Dict, Set

from . import ops
from . import variables


class LineFunction:
    data_variables: List[variables.Data]
    parameter_variables: List[variables.Param]
    index_use_variables: List[variables.IndexUse]
    line: ops.Expr
    data_variable_names: List[str]
    parameter_variable_names: List[str]
    index_use_numpy: List[numpy.array] = []

    def __init__(
        self,
        data_variables: Iterable[str],
        parameter_variables: Iterable[str],
        index_use_variables: Iterable[variables.IndexUse],
        line: ops.Expr,
    ):
        self.data_variables = data_variables
        self.parameter_variables = parameter_variables
        self.data_variable_names = [data.name for data in data_variables]
        self.parameter_variable_names = [
            parameter.name for parameter in parameter_variables
        ]
        self.index_use_variables = list(index_use_variables)
        self.line = line

        vectorize_arguments = (
            [0] * len(self.data_variables)
            + [None] * len(self.parameter_variables)
            + [0] * len(self.index_use_variables)
        )
        function_local_scope = {}
        exec(self.code(), globals(), function_local_scope)
        compiled_function = function_local_scope["func"]
        if any(x is not None for x in vectorize_arguments):
            compiled_function = jax.vmap(compiled_function, vectorize_arguments, 0)

        self.func = lambda *args: jnp.sum(compiled_function(*args))

        self.index_use_numpy = [
            index_use.to_numpy() for index_use in self.index_use_variables
        ]

    def code(self):
        argument_variables = (
            self.data_variables + self.parameter_variables + self.index_use_variables
        )
        args = [variable.code() for variable in argument_variables]
        return "\n".join(
            [f"def func({','.join(args)}):", f"  return {self.line.code()}"]
        )

    def __call__(self, *args):
        return self.func(*args, *self.index_use_numpy)


def compile(data_df: pandas.DataFrame, parsed_lines: List[ops.Expr]):
    data_variables: Dict[str, variables.Data] = {}
    parameter_variables: Dict[str, variables.Param] = {}
    index_variables: Dict[tuple, variables.Index] = {}

    line_functions: List[LineFunction] = []

    for line in parsed_lines:
        assert isinstance(line, ops.Distr)
        data_variables_used: Set[str] = set()
        parameter_variables_used: Set[str] = set()
        index_use_variables: List[variables.IndexUse] = []

        if isinstance(line.variate, ops.Data):
            # If the left hand side is data, the dataframe comes from input
            line_df = data_df
        elif isinstance(line.variate, ops.Param):
            # Otherwise, the dataframe comes from the parameter (unless it's scalar then it's none)
            parameter = parameter_variables[line.variate.get_key()]
            index = parameter.index
            parameter.set_constraints(line.variate.lower, line.variate.upper)

            if index is not None:
                line_df = index.base_df.copy()
                # Rename columns to match names given on the lhs
                if line.variate.index is not None:
                    line_df.columns = line.variate.index.get_key()
            else:
                line_df = None
        else:
            raise Exception(
                f"The left hand side of sampling distribution must be an ops.Data or ops.Param, found {type(line.variate)}"
            )

        for data in ops.search_tree(ops.Data, line):
            data_key = data.get_key()
            data_variables_used.add(data_key)
            if data_key not in data_variables:
                data_variables[data_key] = variables.Data(data_key, line_df[data_key])
            data.variable = data_variables[data_key]

        parameter_index_keys: Dict[str, List[variables.Index]] = {}
        # Find all the ways that each parameter is indexed
        for parameter in ops.search_tree(ops.Param, line):
            parameter_key = parameter.get_key()
            parameter_variables_used.add(parameter_key)

            # Only define new parameters if the parameter is on the right hand side
            if parameter == line.variate:
                continue

            if parameter_key not in parameter_index_keys:
                parameter_index_keys[parameter_key] = []

            if parameter.index is None:
                parameter_index_keys[parameter_key].append(None)
            else:
                parameter_index_keys[parameter_key].append(parameter.index.get_key())

        # Build the parameters
        for parameter_key, index_key_list in parameter_index_keys.items():
            any_none = any(key is None for key in index_key_list)
            all_none = all(key is None for key in index_key_list)
            if any_none:
                # scalar parameters have None has the index
                if all_none:
                    parameter_variables[parameter_key] = variables.Param(parameter_key)
                else:
                    raise Exception("Scalar parameters don't support indexing")
            else:
                columns = list(index_key_list[0])
                value_dfs = []
                for index_key in index_key_list:
                    value_df = line_df.loc[:, index_key]
                    value_df.columns = columns  # columns must be the same to concat
                    value_dfs.append(value_df)

                values_df = pandas.concat(value_dfs, ignore_index=True)
                index = variables.Index(values_df)
                index_variables[parameter_key] = index
                parameter_variables[parameter_key] = variables.Param(
                    parameter_key, index
                )

        for parameter in ops.search_tree(ops.Param, line):
            parameter_key = parameter.get_key()
            if parameter.index is not None:
                index_key = parameter.index.get_key()
                index = index_variables[parameter_key]
                index.incorporate_shifts(
                    parameter.index.shift_columns, parameter.index.shift
                )
                index_df = line_df.loc[:, parameter.index.get_key()]
                index_use_variable = variables.IndexUse(
                    index_key,
                    index_df,
                    index,
                    parameter.index.shift_columns,
                    parameter.index.shift,
                )
                index_use_variables.append(index_use_variable)
                parameter.index.variable = index_use_variable
            parameter.variable = parameter_variables[parameter_key]

        # For each source line, create a python function for log density
        # This will copy over index arrays to jax device
        line_function = LineFunction(
            [data_variables[name] for name in data_variables_used],
            [parameter_variables[name] for name in parameter_variables_used],
            index_use_variables,
            line,
        )

        line_functions.append(line_function)

    return data_variables, parameter_variables, index_variables, line_functions
