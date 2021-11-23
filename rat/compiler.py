import numpy
import pandas
import jax
import jax.numpy as jnp
import jax.scipy.stats
from typing import Iterable, List, Dict, Set

from . import ops
from . import variables


def bernoulli_logit(y, logit_p):
    log_p = -jax.numpy.log1p(jax.numpy.exp(-logit_p))
    log1m_p = -logit_p + log_p
    return jax.numpy.where(y == 0, log1m_p, log_p)


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
        self.parameter_variable_names = [parameter.name for parameter in parameter_variables]
        self.index_use_variables = list(index_use_variables)
        self.line = line

        vectorize_arguments = [0] * len(self.data_variables) + [None] * len(self.parameter_variables) + [0] * len(self.index_use_variables)
        function_local_scope = {}
        exec(self.code(), globals(), function_local_scope)
        compiled_function = function_local_scope["func"]
        if any(x is not None for x in vectorize_arguments):
            compiled_function = jax.vmap(compiled_function, vectorize_arguments, 0)

        self.func = lambda *args: jnp.sum(compiled_function(*args))

        self.index_use_numpy = [index_use.to_numpy() for index_use in self.index_use_variables]

    def code(self):
        argument_variables = self.data_variables + self.parameter_variables + self.index_use_variables
        args = [variable.code() for variable in argument_variables]
        return "\n".join([f"def func({','.join(args)}):", f"  return {self.line.code()}"])

    def __call__(self, *args):
        print("--------")
        for arg in args:
            print(arg.shape)
        for val in self.index_use_variables:
            print(val.names, val.to_numpy().shape)
        print(self.code())
        return self.func(*args, *self.index_use_numpy)


def compile(data_df: pandas.DataFrame, parsed_lines: List[ops.Expr]):

    dependency_graph: Dict[str, Set[str]] = {}
    """the dependency graph stores for a key variable x values as variables that we need to know to evaluate.
    If we think of it as rhs | lhs, the dependency graph is an *acyclic* graph that has directed edges going 
    from rhs -> lhs. This is because subscripts on rhs can be resolved, given the dataframe of the lhs."""

    # traverse through all lines, assuming they are DAGs. This throws operational semantics out the window, but since
    # rat doesn't allow control flow(yet), I'm certain a cycle would mean that it's a mis-specified program

    # Iterate over the rhs variables, and store them in the graph
    for line in parsed_lines:
        if isinstance(line, ops.Distr):
            lhs = line.variate
        elif isinstance(line, ops.Assignment):
            lhs = line.lhs

        lhs_var_key = lhs.get_key()
        if lhs_var_key not in dependency_graph:
            dependency_graph[lhs_var_key] = set()

        for subexpr in ops.search_tree(line, ops.Param, ops.Data):
            rhs_param_key = subexpr.get_key()
            if rhs_param_key not in dependency_graph:
                dependency_graph[rhs_param_key] = set()
            if subexpr.get_key() == lhs_var_key:
                continue

            dependency_graph[rhs_param_key].add(lhs_var_key)

    # evaluation_order is the list of topologically sorted vertices
    evaluation_order = []
    topsort_visited = set()

    def recursive_order_search(current):
        topsort_visited.add(current)
        for child in dependency_graph[current]:
            if child not in topsort_visited:
                recursive_order_search(child)

        evaluation_order.append(current)

    for val in tuple(dependency_graph.keys()):
        if val not in topsort_visited:
            recursive_order_search(val)

    print(evaluation_order)

    # this is the dataframe that goes into variables.Index. Each row is a unique combination of indexes
    parameter_base_df: Dict[str, pandas.DataFrame] = {}

    # this is the dataframe that goes into variables.IndexUse
    parameter_subscripted_df: Dict[str, pandas.DataFrame] = {}

    # this set keep track of variable names that are not parameters, i.e. assigned by assignment
    assigned_parameter_keys: Set[str] = set()

    data_variables: Dict[str, variables.Data] = {}
    parameter_variables: Dict[str, variables.Param] = {}
    assigned_parameter_variables: Dict[str, variables.AssignedParam] = {}
    variable_indexes: Dict[str, variables.Index] = {}

    line_functions: List[LineFunction] = []

    # first pass : fill param_base_dfs of all parameters
    for target_var_name in evaluation_order:
        print("------starting for", target_var_name)
        for line in parsed_lines:
            if isinstance(line, ops.Distr):
                lhs = line.variate
            elif isinstance(line, ops.Assignment):
                lhs = line.lhs
                assigned_parameter_keys.add(lhs.get_key())
            if isinstance(lhs, ops.Data):
                # If the left hand side is data, the dataframe comes from input
                line_df = data_df
                if target_var_name == lhs.get_key():
                    data_variables[lhs.get_key()] = variables.Data(lhs.get_key(), data_df[lhs.get_key()])
            elif isinstance(lhs, ops.Param):
                # Otherwise, the dataframe comes from the parameter (unless it's scalar then it's none)
                if lhs.get_key() in parameter_subscripted_df:
                    if parameter_subscripted_df[lhs.get_key()] is not None:
                        line_df = parameter_subscripted_df[lhs.get_key()].copy()
                        line_df.columns = lhs.index.get_key()
                    else:
                        line_df = None
                else:
                    continue

            parameter_index_keys = []
            # Find all the ways that each parameter is indexed
            for parameter in ops.search_tree(line, ops.Param):
                if parameter == lhs:
                    continue
                parameter_key = parameter.get_key()
                if parameter_key == target_var_name:
                    if parameter.index is None:
                        parameter_index_keys.append(None)
                    else:
                        parameter_index_keys.append(parameter.index.get_key())

            if not parameter_index_keys:
                continue

            any_none = any(key is None for key in parameter_index_keys)
            if not any_none:
                columns = list(parameter_index_keys[0])
                value_dfs = []
                for index_key in parameter_index_keys:
                    value_df = line_df.loc[:, index_key]
                    value_df.columns = columns  # columns must be the same to concat
                    value_dfs.append(value_df)

                # parameter_subscripted_df[target_var_name] = pandas.concat(value_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
                # parameter_base_df[target_var_name] = line_df.loc[:,tuple(parameter_subscripted_df[target_var_name].columns)]

                subscripted_df = pandas.concat(value_dfs,ignore_index=True).drop_duplicates().reset_index(drop=True)

                parameter_subscripted_df[target_var_name] = pandas.concat(value_dfs,ignore_index=True).reset_index(drop=True)
                parameter_base_df[target_var_name] = line_df.loc[:, tuple(subscripted_df.columns)].drop_duplicates().reset_index(drop=True)
                print("base_df:")
                print(parameter_base_df[target_var_name])
                print("subscripted_df:")
                print(parameter_subscripted_df[target_var_name])
                variable_indexes[target_var_name] = variables.Index(parameter_base_df[target_var_name])

                # since parameter subscripts can't change, we can break after filling once
                break

    # print("now printing")
    # for key, val in parameter_base_df.items():
    #     print(key)
    #     print(val)
    # print("done printing")

    """Now transpose the dependency graph(lhs -> rhs, or lhs | rhs), which means traversing the graph in reverse 
    evaluation order. Because we're not trying to evaluate lhs given rhs, this is the canonical dependency graph
    representation for DAGs"""
    evaluation_order = evaluation_order[::-1]

    # since the statements are topologically sorted, a KeyError means a variable was undefined by the user.
    # we now iterate over the statements, assigning actual variables with variables.X
    for target_var_name in evaluation_order:  # current parameter/data name
        for line in parsed_lines:
            # the sets below denote variables needed in this line
            data_vars_used: Set[variables.Data] = set()
            parameter_vars_used: Set[variables.Param] = set()
            assigned_parameter_vars_used: Set[variables.AssignedParam] = set()
            index_use_vars_used: List[variables.IndexUse] = []

            if isinstance(line, ops.Distr):
                lhs = line.variate
            elif isinstance(line, ops.Assignment):
                lhs = line.lhs
            lhs_key = lhs.get_key()
            if lhs_key != target_var_name:
                continue  # worst case we run n! times where n is # of lines

            if isinstance(line, ops.Distr):
                if isinstance(lhs, ops.Data):
                    data_vars_used.add(lhs_key)
                    data_variables[lhs_key] = data_variables[lhs_key]
                    lhs.variable = data_variables[lhs.get_key()]
                elif isinstance(lhs, ops.Param):
                    lower = lhs.lower
                    upper = lhs.upper
                    assert isinstance(lower, ops.RealConstant)
                    assert isinstance(upper, ops.RealConstant)

                    parameter = variables.Param(lhs_key)
                    parameter.set_constraints(lower.value, upper.value)
                    parameter_variables[lhs_key] = parameter
                    lhs.variable = parameter_variables[lhs_key]

            elif isinstance(line, ops.Assignment):
                assert isinstance(line.lhs, ops.Param), "lhs of assignment must be an Identifier denoting a variable name"
                # assignment lhs are set as variables.AssignedParam, since they're not subject for sampling

                parameter = variables.AssignedParam(lhs_key, line.rhs)
                assigned_parameter_variables[lhs_key] = parameter

            # once variable/parameter lhs declarations/sampling are handled, we process rhs
            for var in ops.search_tree(line, ops.Data):
                var_key = var.get_key()
                data_vars_used.add(var_key)
                if var_key == lhs_key:
                    continue
                data_variables[var_key] = data_variables[var_key]
                var.variable = data_variables[var_key]

            for var in ops.search_tree(line, ops.Param):
                var_key = var.get_key()
                if var_key in assigned_parameter_variables:
                    assigned_parameter_vars_used.add(var_key)
                else:
                    parameter_vars_used.add(var_key)
                # if var_key == lhs_key:
                #     continue

                if var_key in assigned_parameter_variables:
                    assigned_parameter_vars_used.add(var_key)
                    var.variable = assigned_parameter_variables[var_key]
                else:
                    parameter_vars_used.add(var_key)
                    var.variable = parameter_variables[var_key]
                if var.index:
                    index_use = variables.IndexUse(
                        var.index.get_key(), parameter_subscripted_df[lhs.get_key()], variable_indexes[var_key], var.index.shifts
                    )
                    variable_indexes[var_key].incorporate_shifts(var.index.shifts)
                    var.index.variable = index_use
                    index_use_vars_used.append(index_use)
                    print("appended", var_key)

            line_function = LineFunction(
                [data_variables[name] for name in data_vars_used],
                [parameter_variables[name] for name in parameter_vars_used],
                index_use_vars_used,
                line,
            )
            line_functions.append(line_function)
            print("-------------------")
    return data_variables, parameter_variables, assigned_parameter_variables, variable_indexes, line_functions
