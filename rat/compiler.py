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
    index_variables: Dict[str, variables.Index] = {}
    indexuse_variables: List[variables.IndexUse] = []
    assigned_parameter_variables: Dict[str, variables.AssignedVariable] = {}

    variable_index_keys: Dict[str, Tuple[str]] = defaultdict(lambda : None)
    variable_index_df: Dict[str, pandas.DataFrame] = {}


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

    # second pass - look at rhs and extract index dataframes
    for line in parsed_lines:
        for parameter in ops.search_tree(ops.Param, line):
            if isinstance(line, ops.Distr):
                if parameter == line.variate:
                    continue
            elif isinstance(line, ops.Assignment):
                if parameter == line.lhs:
                    continue

            parameter_key = parameter.get_key()
            if parameter.index:
                index_key = tuple(parameter.index.get_key())

                value_df = data_df.loc[:, tuple(parameter.index.get_key())]
                if isinstance(value_df, pandas.Series): value_df = value_df.to_frame()
                if parameter_key not in variable_index_df:
                    variable_index_df[parameter_key] = value_df
                else:
                    value_df.columns = variable_index_df[parameter_key].columns
                    variable_index_df[parameter_key] = pandas.concat([variable_index_df[parameter_key], value_df], ignore_index=True)


    for variable_name, unprocessed_df in variable_index_df.items():
        index_variables[variable_name] = variables.Index(unprocessed_df)


    for line in parsed_lines:
        for parameter in ops.search_tree(ops.Param, line):

            parameter_key = parameter.get_key()
            if parameter.index:
                print(parameter.index)
                index_key = tuple(parameter.index.get_key())
                value_df = data_df.loc[:, index_key]
                if parameter_key not in index_variables:
                    raise Exception(f"Subscript mismatch error - parameter '{parameter_key}' is being used with and without subscripts.")
                var_index = index_variables[parameter_key]
                index_use_variable = variables.IndexUse(
                    index_key,
                    value_df,
                    var_index,
                    parameter.index.shifts,
                )
                if(index_key not in [ik.names for ik in indexuse_variables]):
                    indexuse_variables.append(index_use_variable)
                if parameter_key in parameter_variables:
                    parameter_variables[parameter_key].index = var_index
                else:
                    assigned_parameter_variables[parameter_key].index = var_index

                parameter.index.variable = index_use_variable

            else:
                if parameter_key in index_variables:
                    raise Exception(f"Subscript mismatch error - parameter '{parameter_key}' is being used with and without subscripts.")

            if parameter_key in parameter_variables:
                parameter.variable = parameter_variables[parameter_key]
            else:
                parameter.variable = assigned_parameter_variables[parameter_key]


    # generate function for each line
    for line in parsed_lines:
        if(isinstance(line, ops.Distr)):
            data_variables_used: Set[str] = set()
            parameter_variables_used: Set[str] = set()
            assigned_parameter_variables_used: Set[str] = set()
            index_key_used = set()

            for data in ops.search_tree(ops.Data, line):
                data_variables_used.add(data.get_key())

            for param in ops.search_tree(ops.Param, line):
                param_key = param.get_key()
                if param.index.get_key():
                    index_key_used.add(param.index.get_key())
                if param_key in assigned_parameter_variables:
                    assigned_parameter_variables_used.add(param_key)
                else:
                    parameter_variables_used.add(param_key)
            line_function = LineFunction(
                [data_variables[name] for name in data_variables_used],
                [parameter_variables[name] for name in parameter_variables_used],
                [indexuse for indexuse in indexuse_variables if indexuse.names in index_key_used],
                [assigned_parameter_variables[name] for name in assigned_parameter_variables_used],
                line,
            )
            line_functions.append(line_function)

    return data_variables, parameter_variables, index_variables, assigned_parameter_variables, line_functions

