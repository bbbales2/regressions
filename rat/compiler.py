import logging
import numpy
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import Iterable, List, Dict, Set, Tuple

from . import ops
from . import variables


def bernoulli_logit(y, logit_p):
    log_p = -jax.numpy.log1p(jax.numpy.exp(-logit_p))
    log1m_p = -logit_p + log_p
    return jax.numpy.where(y == 0, log1m_p, log_p)


def log_normal(y, mu, sigma):
    logy = jax.numpy.log(y)
    return jax.scipy.stats.norm.logpdf(logy, mu, sigma) - logy


class LineFunction:
    data_variables: List[variables.Data]
    parameter_variables: List[variables.Param]
    index_use_variables: List[variables.IndexUse]
    assigned_parameter_variables: List[variables.AssignedParam]
    line: ops.Expr
    data_variable_names: List[str]
    parameter_variable_names: List[str]
    index_use_numpy: List[numpy.array]

    def __init__(
        self,
        data_variables: Iterable[variables.Data],
        parameter_variables: Iterable[variables.Param],
        assigned_parameter_variables: Iterable[variables.AssignedParam],
        index_use_variables: Iterable[variables.IndexUse],
        line: ops.Expr,
    ):
        self.data_variables = data_variables
        self.parameter_variables = parameter_variables
        self.data_variable_names = [data.name for data in data_variables]
        self.parameter_variable_names = [parameter.name for parameter in parameter_variables]
        self.assigned_parameter_variables = assigned_parameter_variables
        self.assigned_parameter_variables_names = [parameter.ops_param.name for parameter in assigned_parameter_variables]
        self.index_use_variables = list(index_use_variables)
        self.line = line

        vectorize_arguments = (
            [0] * len(self.data_variables)
            + [None] * len(self.parameter_variables)
            + [None] * len(self.assigned_parameter_variables)
            + [0] * len(self.index_use_variables)
        )
        function_local_scope = {}
        exec(self.code(), globals(), function_local_scope)
        self.compiled_function = function_local_scope["func"]
        if any(x is not None for x in vectorize_arguments):
            self.compiled_function = jax.vmap(self.compiled_function, vectorize_arguments, 0)

        self.index_use_numpy = [index_use.to_numpy() for index_use in self.index_use_variables]

    def code(self):
        argument_variables = self.data_variables + self.parameter_variables + self.assigned_parameter_variables + self.index_use_variables
        args = [variable.code() for variable in argument_variables]
        return "\n".join([f"def func({','.join(args)}):", f"  return {self.line.code()}"])

    def compiled_function_sum(self, *args):
        return jax.numpy.sum(self.compiled_function(*args))

    def __call__(self, *args):
        return self.compiled_function_sum(*args, *self.index_use_numpy)


class AssignLineFunction(LineFunction):
    name: str

    def __init__(
        self,
        name: str,
        data_variables: Iterable[variables.Data],
        parameter_variables: Iterable[variables.Param],
        assigned_parameter_variables: Iterable[variables.AssignedParam],
        index_use_variables: Iterable[variables.IndexUse],
        line: ops.Expr,
    ):
        self.name = name
        super().__init__(data_variables, parameter_variables, assigned_parameter_variables, index_use_variables, line)

    def code(self):
        argument_variables = self.data_variables + self.parameter_variables + self.assigned_parameter_variables + self.index_use_variables
        args = [variable.code() for variable in argument_variables]
        return "\n".join([f"def func({','.join(args)}):", f"  return {self.line.rhs.code()}"])

    def __call__(self, *args):
        return self.compiled_function(*args, *self.index_use_numpy)


# Iterate over the rhs variables, and store them in the graph
def generate_dependency_graph(parsed_lines: List[ops.Expr], reversed: bool = True):
    out_graph = {}
    for line in parsed_lines:
        if isinstance(line, ops.Distr):
            lhs = line.variate
        elif isinstance(line, ops.Assignment):
            lhs = line.lhs

        lhs_op_key = lhs.get_key()
        if lhs_op_key not in out_graph:
            out_graph[lhs_op_key] = set()

        for subexpr in ops.search_tree(line, ops.Param, ops.Data):
            rhs_op_key = subexpr.get_key()
            if rhs_op_key not in out_graph:
                out_graph[rhs_op_key] = set()
            if subexpr.get_key() == lhs_op_key:
                continue

            if reversed:
                out_graph[rhs_op_key].add(lhs_op_key)
            else:
                # if not isinstance(lhs, ops.Data):  # Data is independent, need nothing to evaluate data
                out_graph[lhs_op_key].add(rhs_op_key)
            # if isinstance(line, ops.Distr):
            #     if reversed:
            #         out_graph[rhs_op_key].add(lhs_op_key)
            #     else:
            #         # if not isinstance(lhs, ops.Data):  # Data is independent, need nothing to evaluate data
            #         out_graph[lhs_op_key].add(rhs_op_key)
            # else:
            #     if reversed:
            #         out_graph[rhs_op_key].add(lhs_op_key)
            #     else:
            #         out_graph[lhs_op_key].add(rhs_op_key)

    return out_graph


def topological_sort(graph):
    evaluation_order = []
    topsort_visited = set()

    def recursive_order_search(current):
        topsort_visited.add(current)
        for child in graph[current]:
            if child not in topsort_visited:
                recursive_order_search(child)

        evaluation_order.append(current)

    for val in tuple(graph.keys()):
        if val not in topsort_visited:
            recursive_order_search(val)

    return evaluation_order


def compile(data_df: pandas.DataFrame, parsed_lines: List[ops.Expr]):
    logging.info(f"rat compiler - starting code generation for {len(parsed_lines)} line(s) of code.")
    dependency_graph: Dict[str, Set[str]] = {}
    """the dependency graph stores for a key variable x values as variables that we need to know to evaluate.
    If we think of it as rhs | lhs, the dependency graph is an *acyclic* graph that has directed edges going 
    from rhs -> lhs. This is because subscripts on rhs can be resolved, given the dataframe of the lhs. However, for
    assignments, the order is reversed(lhs -> rhs) since we need rhs to infer its subscripts"""

    # traverse through all lines, assuming they are DAGs. This throws operational semantics out the window, but since
    # rat doesn't allow control flow(yet), I'm certain a cycle would mean that it's a mis-specified program

    dependency_graph = generate_dependency_graph(parsed_lines)
    evaluation_order = topological_sort(dependency_graph)
    # assignments need to be treated opposite

    # evaluation_order is the list of topologically sorted vertices

    logging.debug(f"reverse dependency graph: {dependency_graph}")
    logging.debug(f"first pass eval order: {evaluation_order}")

    # this is the dataframe that goes into variables.Index. Each row is a unique combination of indexes
    parameter_base_df: Dict[str, pandas.DataFrame] = {}

    # this is the dictionary that keeps track of what columns the parameter subscripts were defined as
    parameter_index_columns: Dict[str, List[Set[str]]] = {}

    # this set keep track of variable names that are not parameters, i.e. assigned by assignment
    assigned_parameter_keys: Set[str] = set()

    data_variables: Dict[str, variables.Data] = {}
    parameter_variables: Dict[str, variables.Param] = {}
    assigned_parameter_variables: Dict[str, variables.AssignedParam] = {}
    variable_indexes: Dict[str, variables.Index] = {}

    line_functions: List[LineFunction] = []

    # first pass : fill param_base_dfs of all parameters
    for target_op_name in evaluation_order:
        logging.debug(f"------first pass starting for {target_op_name}")
        target_op_index_key = None
        for line in parsed_lines:
            if isinstance(line, ops.Distr):
                lhs = line.variate
            elif isinstance(line, ops.Assignment):
                lhs = line.lhs
                assigned_parameter_keys.add(lhs.get_key())
                if lhs.get_key() == target_op_name:
                    if lhs.index:
                        if lhs.get_key() not in parameter_base_df:
                            raise Exception(f"Can't resolve {lhs.get_key()}")
                        target_op_index_key = lhs.index.get_key()

            if not lhs.get_key() == target_op_name:
                continue

            line_df = None
            if isinstance(lhs, ops.Data):
                # If the left hand side is data, the dataframe comes from input
                line_df = data_df
            elif isinstance(lhs, ops.Param):
                # Otherwise, the dataframe comes from the parameter (unless it's scalar then it's none)
                if lhs.get_key() in parameter_base_df:
                    if parameter_base_df[lhs.get_key()] is not None:
                        if lhs.index is None:
                            raise Exception(f"{lhs.get_key()} must declare its subscripts to be used as a reference variable")
                        target_op_index_key = lhs.index.get_key()
                        line_df = parameter_base_df[lhs.get_key()].copy()
                        line_df.columns = tuple(lhs.index.get_key())
                    else:
                        line_df = None
                else:
                    if not lhs.index:
                        break

            # Find all the data referenced in the line
            for subexpr in ops.search_tree(line, ops.Data):
                data_variables[subexpr.get_key()] = variables.Data(subexpr.get_key(), data_df[subexpr.get_key()])

            # Find all the ways that each parameter is indexed
            for subexpr in ops.search_tree(line, ops.Param):
                # Is this handling y = y by ignoring it?
                if subexpr.get_key() == lhs.get_key():
                    continue
                subexpr_key = subexpr.get_key()
                if subexpr.index:
                    try:
                        v_df = line_df.loc[:, subexpr.index.get_key()]
                    except KeyError as e:
                        raise Exception(f"Dataframe for {lhs.get_key()} cannot be subscripted by {subexpr.index.get_key()}") from e
                    logging.debug(f"{subexpr_key} referenced")
                    if subexpr_key in parameter_base_df:
                        v_df.columns = tuple(parameter_base_df[subexpr_key].columns)
                        parameter_base_df[subexpr_key] = (
                            pandas.concat([parameter_base_df[subexpr_key], v_df]).drop_duplicates().reset_index(drop=True)
                        )
                        for n, key in enumerate(subexpr.index.get_key()):
                            parameter_index_columns[subexpr_key][n].add(key)
                    else:
                        # initialize the index column list length to be the index count
                        parameter_index_columns[subexpr_key] = [set() for _ in range(len(subexpr.index.get_key()))]
                        for n, key in enumerate(subexpr.index.get_key()):
                            parameter_index_columns[subexpr_key][n].add(key)
                        parameter_base_df[subexpr_key] = v_df.drop_duplicates().reset_index(drop=True)

        # else:
        #     # Ignore variables in the input dataframe
        #     if target_op_name not in data_df.columns:
        #         raise Exception(
        #             f"Could not find a definition for {target_op_name} (it should either have a prior if it's a parameter or be assigned if it's a transformed parameter)"
        #         )
        if target_op_name in parameter_base_df:
            parameter_base_df[target_op_name].columns = tuple(target_op_index_key)

    logging.debug("first pass finished")
    logging.debug("now printing parameter_base_df")
    for key, val in parameter_base_df.items():
        logging.debug(key)
        logging.debug(val)
        variable_indexes[key] = variables.Index(parameter_base_df[key], parameter_index_columns[key])
    logging.debug("done printing parameter_base_df")
    logging.debug("now printing variable_indexes")
    for key, val in variable_indexes.items():
        logging.debug(key)
        logging.debug(val.base_df)
        logging.info(f"Subscript data for parameter {key}:")
        val.log_summary(logging.INFO)
        logging.info("-----")
    logging.debug("done printing variable_indexes")

    """Now transpose the dependency graph(lhs -> rhs, or lhs | rhs). For a normal DAD, it's as simple as reversing the
     evaluation order. Unfortunately, since assignments always stay true as (lhs | rhs), we have to rebuild the DAG. :(
     Because we're trying to evaluate lhs given rhs, this is the canonical dependency graph representation for DAGs"""
    dependency_graph = generate_dependency_graph(parsed_lines, reversed=False)
    evaluation_order = topological_sort(dependency_graph)
    logging.debug(f"canonical dependency graph: {dependency_graph}")
    logging.debug(f"second pass eval order: {evaluation_order}")

    # since the statements are topologically sorted, a KeyError means a variable was undefined by the user.
    # we now iterate over the statements, assigning actual variables with variables.X
    for target_op_name in evaluation_order:  # current parameter/data name
        logging.debug("@" * 5)
        logging.debug(f"running 2nd pass for: {target_op_name}")
        logging.debug(f"data: {list(data_variables.keys())}")
        logging.debug(f"parameters: {list(parameter_variables.keys())}")
        logging.debug(f"assigned parameters {list(assigned_parameter_variables.keys())}")

        for line in parsed_lines:
            # the sets below denote variables needed in this line
            data_vars_used: Set[variables.Data] = set()
            parameter_vars_used: Set[variables.Param] = set()
            assigned_parameter_vars_used: Set[variables.AssignedParam] = set()
            index_use_vars: Dict[Tuple, variables.IndexUse] = {}

            if isinstance(line, ops.Distr):
                lhs = line.variate
            elif isinstance(line, ops.Assignment):
                lhs = line.lhs
            lhs_key = lhs.get_key()
            if lhs_key != target_op_name:
                continue  # worst case we run n! times where n is # of lines

            if isinstance(lhs, ops.Data):
                lhs_df = data_df
            else:
                if lhs.index is not None:
                    lhs_df = variable_indexes[lhs.get_key()].df  # parameter_base_df[lhs.get_key()]

            if isinstance(line, ops.Distr):
                if isinstance(lhs, ops.Data):
                    data_vars_used.add(lhs_key)
                    lhs.variable = data_variables[lhs.get_key()]
                elif isinstance(lhs, ops.Param):
                    lower = lhs.lower
                    upper = lhs.upper
                    assert isinstance(lower, ops.RealConstant)
                    assert isinstance(upper, ops.RealConstant)

                    if lhs_key in variable_indexes:
                        subexpr = variables.Param(lhs_key, variable_indexes[lhs_key])
                    else:
                        subexpr = variables.Param(lhs_key)
                    subexpr.set_constraints(lower.value, upper.value)
                    parameter_variables[lhs_key] = subexpr
                    lhs.variable = parameter_variables[lhs_key]

            elif isinstance(line, ops.Assignment):
                assert isinstance(line.lhs, ops.Param), "lhs of assignment must be an Identifier denoting a variable name"
                # assignment lhs are set as variables.AssignedParam, since they're not subject for sampling

                if lhs.index:
                    subexpr = variables.AssignedParam(lhs, line.rhs, variable_indexes[lhs_key])
                    variable_indexes[lhs.get_key()].incorporate_shifts(lhs.index.shifts)
                    subexpr.ops_param.index.variable = variables.IndexUse(
                        lhs.index.get_key(), parameter_base_df[lhs.get_key()], variable_indexes[lhs_key]
                    )
                else:
                    subexpr = variables.AssignedParam(lhs, line.rhs)
                assigned_parameter_variables[lhs_key] = subexpr
                lhs.variable = subexpr

            # once variable/parameter lhs declarations/sampling are handled, we process rhs
            for op in ops.search_tree(line, ops.Data):
                op_key = op.get_key()
                data_vars_used.add(op_key)
                if op_key == lhs_key:
                    continue
                op.variable = data_variables[op_key]

            for op in ops.search_tree(line, ops.Param):
                op_key = op.get_key()

                # For assignments, the left hand isn't defined yet so don't mark it as used
                if isinstance(line, ops.Assignment) and op_key == lhs_key:
                    continue

                if op_key in assigned_parameter_variables:
                    assigned_parameter_vars_used.add(op_key)
                else:
                    parameter_vars_used.add(op_key)

                if op_key in assigned_parameter_variables:
                    assigned_parameter_vars_used.add(op_key)
                    op.variable = assigned_parameter_variables[op_key]
                else:
                    parameter_vars_used.add(op_key)
                    op.variable = parameter_variables[op_key]

                if op.index:
                    variable_indexes[op_key].incorporate_shifts(op.index.shifts)

                    index_use_identifier = (op.index.get_key(), op.index.shifts)

                    df_index = (
                        op.index.get_key() if isinstance(lhs, ops.Data) else lhs.variable.index.check_and_return_index(op.index.get_key())
                    )

                    # Only need to define this particular index use once per line (multiple uses will share)
                    if index_use_identifier not in index_use_vars:
                        index_use = variables.IndexUse(
                            op.index.get_key(), lhs_df.loc[:, df_index], variable_indexes[op_key], op.index.shifts
                        )

                        index_use_vars[index_use_identifier] = index_use

                    op.index.variable = index_use_vars[index_use_identifier]

            for op in ops.search_tree(line, ops.Data):
                op_key = op.get_key()
                if op.index:
                    raise Exception(f"Indexing on data variables ({op_key}) not supported")

            if isinstance(line, ops.Distr):
                line_function = LineFunction(
                    [data_variables[name] for name in data_vars_used],
                    [parameter_variables[name] for name in parameter_vars_used],
                    [assigned_parameter_variables[name] for name in assigned_parameter_vars_used],
                    index_use_vars.values(),
                    line,
                )
            elif isinstance(line, ops.Assignment):
                line_function = AssignLineFunction(
                    lhs_key,
                    [data_variables[name] for name in data_vars_used],
                    [parameter_variables[name] for name in parameter_vars_used],
                    [assigned_parameter_variables[name] for name in assigned_parameter_vars_used],
                    index_use_vars.values(),
                    line,
                )
            line_functions.append(line_function)
            logging.debug("-------------------")
            break

    logging.debug("2nd pass finished")
    logging.debug(f"data: {list(data_variables.keys())}")
    logging.debug(f"parameters: {list(parameter_variables.keys())}")
    logging.debug(f"assigned parameters {list(assigned_parameter_variables.keys())}")
    return data_variables, parameter_variables, assigned_parameter_variables, variable_indexes, line_functions
