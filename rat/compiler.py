import numpy
import pandas
import jax
import jax.numpy as jnp
import jax.scipy.stats
from typing import Iterable, List, Dict, Set, Tuple
from collections import OrderedDict, defaultdict
import inspect
from . import ops
from . import variables
import warnings

def bernoulli_logit(y, logit_p):
    log_p = -jax.numpy.log1p(jax.numpy.exp(-logit_p))
    log1m_p = -logit_p + log_p
    return jax.numpy.where(y == 0, log1m_p, log_p)


class LineFunction:
    """
    Generates a python function for the log probability

    Attributes:
        data_variables:
    """
    data_variables: List[variables.Data]
    parameter_variables: List[variables.Param]
    index_use_variables: List[variables.IndexUse]
    assigned_parameter_variables: List[variables.AssignedParam]
    line: ops.Expr
    data_variable_names: List[str]
    parameter_variable_names: List[str]
    assigned_parameter_variable_names: List[str]
    index_use_numpy: List[numpy.array] = []

    def __init__(
        self,
        data_variables: Iterable[variables.Data],
        parameter_variables: Iterable[variables.Param],
        index_use_variables: Iterable[variables.IndexUse],
        assigned_parameter_variables: Iterable[variables.AssignedParam],
        line: ops.Expr,
    ):
        self.data_variables = data_variables
        self.parameter_variables = parameter_variables
        self.data_variable_names = [data.name for data in data_variables]
        self.parameter_variable_names = [
            parameter.name for parameter in parameter_variables
        ]
        self.index_use_variables = list(index_use_variables)
        self.assigned_parameter_variables = assigned_parameter_variables
        self.assigned_parameter_variable_names = [ap.name for ap in assigned_parameter_variables]
        self.line = line

        vectorize_arguments = (
            [0] * len(self.data_variables)
            + [None] * len(self.parameter_variables)
            + [0] * len(self.index_use_variables)
            + [0] * len(self.assigned_parameter_variable_names)
        )

        function_local_scope = {}
        exec(self.code(), globals(), function_local_scope)
        compiled_function = function_local_scope["func"]
        print(vectorize_arguments, self.code())
        if any(x is not None for x in vectorize_arguments):
            compiled_function = jax.vmap(compiled_function, vectorize_arguments, 0)

        self.func = lambda *args: jnp.sum(compiled_function(*args))

        self.index_use_numpy = [
            index_use.to_numpy() for index_use in self.index_use_variables
        ]

    def code(self):
        argument_variables = (
            self.data_variables + self.parameter_variables + self.index_use_variables + self.assigned_parameter_variables
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
    indexuse_variables: List[variables.IndexUse] = []
    assigned_parameter_variables: Dict[str, variables.AssignedVariable] = {}

    variable_index_keys: Dict[str, Tuple[str]] = defaultdict(lambda : None)


    line_functions: List[LineFunction] = []

    # first pass - find lhs param/assigned_params and build empty variable.x for each
    for line in parsed_lines:
        if isinstance(line, ops.Distr):
            if isinstance(line.variate, ops.Param):
                # distribution variates are set as variable.Param
                parameter_key = line.variate.get_key()

                lower = line.variate.lower
                upper = line.variate.upper
                assert isinstance(lower, ops.RealConstant)
                assert isinstance(upper, ops.RealConstant)

                parameter = variables.Param(parameter_key)
                parameter.set_constraints(lower.value, upper.value)
                parameter_variables[parameter_key] = parameter

        elif isinstance(line, ops.Assignment):
            assert isinstance(line.lhs, ops.Param), "lhs of assignemnt must be an Identifier denoting a variable name"
            # assignment lhs are aset as assignedparam, since they're not subject for sampling
            parameter_key = line.lhs.get_key()
            parameter = variables.AssignedParam(parameter_key, line.rhs)
            assigned_parameter_variables[parameter_key] = parameter

        else:
            raise Exception(f"Don't know how to handle type of expression {line.__class__.__name__} in a statement.")

        # add data variables
        for data in ops.search_tree(ops.Data, line):
            data_key = data.get_key()
            data_variables[data_key] = variables.Data(data_key, data_df[data_key])
            data.variable = data_variables[data_key]

    # second pass - assign indexes
    for line in parsed_lines:
        # find how parameters are subscripted
        for parameter in ops.search_tree(ops.Param, line):
            parameter_key = parameter.get_key()
            if parameter.index:
                index_key = tuple(parameter.index.get_key())

                # for assigned_params, check all subscripts match
                if parameter_key in assigned_parameter_variables:
                    for subparams in ops.search_tree(ops.Param, parameter):
                        if subparams.index.get_key() != index_key:
                            raise Exception(
                                f"Subscript mismatch while checking assignment for variable name {parameter.name}: current subscript was {index_key}, but variable {subparams.name} has subscript {subparams.index.get_key()}")

                # for all parameters, check subscripts are consistent across all lines
                if variable_index_keys[parameter_key] and variable_index_keys[parameter_key] != index_key:
                    warnings.warn(f"Subscripts for parameter '{parameter_key}' has already been defined as {variable_index_keys[parameter_key]}, but a different subscript {index_key} was given again. This may be problematic.")

                variable_index_keys[parameter_key] = index_key
                value_df = data_df.loc[:, tuple(parameter.index.get_key())]
                if isinstance(value_df, pandas.Series): value_df = value_df.to_frame()

                index = variables.Index(value_df)
                index.incorporate_shifts(parameter.index.shifts)

                index_variables[index_key] = index

                index_use_variable = variables.IndexUse(
                    index_key,
                    value_df,
                    index,
                    parameter.index.shifts,
                )
                indexuse_variables.append(index_use_variable)
                if parameter_key in parameter_variables:
                    parameter_variables[parameter_key].index = index
                else:
                    assigned_parameter_variables[parameter_key].index = index

                parameter.index.variable = index_use_variable

            if parameter_key in parameter_variables:
                parameter.variable = parameter_variables[parameter_key]
            else:
                parameter.variable = assigned_parameter_variables[parameter_key]



    # third pass - generate function for each line
    for line in parsed_lines:
        if(isinstance(line, ops.Distr)):
            data_variables_used: Set[str] = set()
            parameter_variables_used: Set[str] = set()
            assigned_parameter_variables_used: Set[str] = set()

            for data in ops.search_tree(ops.Data, line):
                data_variables_used.add(data.get_key())

            for param in ops.search_tree(ops.Param, line):
                param_key = param.get_key()
                if param_key in assigned_parameter_variables:
                    assigned_parameter_variables_used.add(param_key)
                else:
                    parameter_variables_used.add(param_key)
            line_function = LineFunction(
                [data_variables[name] for name in data_variables_used],
                [parameter_variables[name] for name in parameter_variables_used],
                indexuse_variables,
                [assigned_parameter_variables[name] for name in assigned_parameter_variables_used],
                line,
            )
            line_functions.append(line_function)

    return data_variables, parameter_variables, index_variables, assigned_parameter_variables, line_functions


# def compile(data_df: pandas.DataFrame, parsed_lines: List[ops.Expr]):
#     data_variables: Dict[str, variables.Data] = {}
#     parameter_variables: Dict[str, variables.Param] = {}
#     index_variables: Dict[tuple, variables.Index] = {}
#
#     line_functions: List[LineFunction] = []
#
#     for line in parsed_lines:
#         print(parameter_variables)
#         assert isinstance(line, ops.Distr)
#         data_variables_used: Set[str] = set()
#         parameter_variables_used: Set[str] = set()
#         index_use_variables: List[variables.IndexUse] = []
#
#         if isinstance(line.variate, ops.Data):
#             # If the left hand side is data, the dataframe comes from input
#             line_df = data_df
#         elif isinstance(line.variate, ops.Param):
#             # Otherwise, the dataframe comes from the parameter (unless it's scalar then it's none)
#             parameter = parameter_variables[line.variate.get_key()]
#             index = parameter.index
#
#             lower = line.variate.lower
#             upper = line.variate.upper
#             assert isinstance(lower, ops.RealConstant)
#             assert isinstance(upper, ops.RealConstant)
#
#             parameter.set_constraints(lower.value, upper.value)
#
#             if index is not None:
#                 line_df = index.base_df.copy()
#                 # Rename columns to match names given on the lhs
#                 if line.variate.index is not None:
#                     line_df.columns = line.variate.index.get_key()
#             else:
#                 line_df = None
#         else:
#             raise Exception(
#                 f"The left hand side of sampling distribution must be an ops.Data or ops.Param, found {type(line.variate)}"
#             )
#
#         for data in ops.search_tree(ops.Data, line):
#             data_key = data.get_key()
#             data_variables_used.add(data_key)
#             if data_key not in data_variables:
#                 data_variables[data_key] = variables.Data(data_key, line_df[data_key])
#             data.variable = data_variables[data_key]
#
#         parameter_index_keys: Dict[str, List[variables.Index]] = {}
#         # Find all the ways that each parameter is indexed
#         for parameter in ops.search_tree(ops.Param, line):
#             parameter_key = parameter.get_key()
#             parameter_variables_used.add(parameter_key)
#
#             # Only define new parameters if the parameter is on the right hand side
#             if parameter == line.variate:
#                 continue
#
#             if parameter_key not in parameter_index_keys:
#                 parameter_index_keys[parameter_key] = []
#
#             if parameter.index is None:
#                 parameter_index_keys[parameter_key].append(None)
#             else:
#                 parameter_index_keys[parameter_key].append(parameter.index.get_key())
#
#         # Build the parameters
#         for parameter_key, index_key_list in parameter_index_keys.items():
#             any_none = any(key is None for key in index_key_list)
#             all_none = all(key is None for key in index_key_list)
#             if any_none:
#                 # scalar parameters have None has the index
#                 if all_none:
#                     parameter_variables[parameter_key] = variables.Param(parameter_key)
#                 else:
#                     raise Exception("Scalar parameters don't support indexing")
#             else:
#                 columns = list(index_key_list[0])
#                 value_dfs = []
#                 for index_key in index_key_list:
#                     value_df = line_df.loc[:, index_key]
#                     value_df.columns = columns  # columns must be the same to concat
#                     value_dfs.append(value_df)
#
#                 values_df = pandas.concat(value_dfs, ignore_index=True)
#                 print(parameter_key, values_df)
#                 index = variables.Index(values_df)
#                 index_variables[parameter_key] = index
#                 parameter_variables[parameter_key] = variables.Param(
#                     parameter_key, index
#                 )
#
#         for parameter in ops.search_tree(ops.Param, line):
#             parameter_key = parameter.get_key()
#             if parameter.index is not None:
#                 index_key = parameter.index.get_key()
#                 index = index_variables[parameter_key]
#                 index.incorporate_shifts(parameter.index.shifts)
#                 index_df = line_df.loc[:, parameter.index.get_key()]
#                 index_use_variable = variables.IndexUse(
#                     index_key,
#                     index_df,
#                     index,
#                     parameter.index.shifts,
#                 )
#                 index_use_variables.append(index_use_variable)
#                 parameter.index.variable = index_use_variable
#             parameter.variable = parameter_variables[parameter_key]
#
#         # For each source line, create a python function for log density
#         # This will copy over index arrays to jax device
#         line_function = LineFunction(
#             [data_variables[name] for name in data_variables_used],
#             [parameter_variables[name] for name in parameter_variables_used],
#             index_use_variables,
#             line,
#         )
#
#         line_functions.append(line_function)
#
#     return data_variables, parameter_variables, index_variables, line_functions
