import logging
import numpy
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import Iterable, List, Dict, Set, Tuple
import warnings

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
    subscript_use_variables: List[variables.SubscriptUse]
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
        subscript_use_variables: Iterable[variables.SubscriptUse],
        line: ops.Expr,
    ):
        self.data_variables = data_variables
        self.parameter_variables = parameter_variables
        self.data_variable_names = [data.name for data in data_variables]
        self.parameter_variable_names = [parameter.name for parameter in parameter_variables]
        self.assigned_parameter_variables = assigned_parameter_variables
        self.assigned_parameter_variables_names = [parameter.ops_param.name for parameter in assigned_parameter_variables]
        self.subscript_use_variables = list(subscript_use_variables)
        self.line = line

        vectorize_arguments = (
            [0] * len(self.data_variables)
            + [None] * len(self.parameter_variables)
            + [None] * len(self.assigned_parameter_variables)
            + [0] * len(self.subscript_use_variables)
        )
        function_local_scope = {}
        exec(self.code(), globals(), function_local_scope)
        self.compiled_function = function_local_scope["func"]
        if any(x is not None for x in vectorize_arguments):
            self.compiled_function = jax.vmap(self.compiled_function, vectorize_arguments, 0)

        self.index_use_numpy = [index_use.to_numpy() for index_use in self.subscript_use_variables]

    def code(self):
        argument_variables = (
            self.data_variables + self.parameter_variables + self.assigned_parameter_variables + self.subscript_use_variables
        )
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
        subscript_use_variables: Iterable[variables.SubscriptUse],
        line: ops.Expr,
    ):
        self.name = name
        super().__init__(data_variables, parameter_variables, assigned_parameter_variables, subscript_use_variables, line)

    def code(self):
        argument_variables = (
            self.data_variables + self.parameter_variables + self.assigned_parameter_variables + self.subscript_use_variables
        )
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


class CompileError(Exception):
    def __init__(self, message, code_string: str, line_num: int, column_num: int):
        code_string = code_string.split("\n")[line_num] if code_string else ""
        exception_message = f"An error occured while compiling the following line({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        super().__init__(exception_message)


class Compiler:
    data_df: pandas.DataFrame
    expr_tree_list: List[ops.Expr]
    model_code_string: str

    def __init__(self, data_df: pandas.DataFrame, expr_tree_list: List[ops.Expr], model_code_string: str = ""):
        self.data_df = data_df
        self.expr_tree_list = expr_tree_list
        self.model_code_string = model_code_string

        # this is the dataframe that goes into variables.Subscript. Each row is a unique combination of indexes
        self.parameter_base_df: Dict[str, pandas.DataFrame] = {}

        # this is the dictionary that keeps track of what columns the parameter subscripts were defined as
        self.parameter_subscript_columns: Dict[str, List[Set[str]]] = {}

        # this set keep track of variable names that are not parameters, i.e. assigned by assignment
        self.assigned_parameter_keys: Set[str] = set()

        self.data_variables: Dict[str, variables.Data] = {}
        self.parameter_variables: Dict[str, variables.Param] = {}
        self.assigned_parameter_variables: Dict[str, variables.AssignedParam] = {}
        self.variable_subscripts: Dict[str, variables.Subscript] = {}

        self.line_functions: List[LineFunction] = []

    def _first_pass(self):
        # On the first pass, we generate the base reference dataframe for each variable. This goes into
        # variable.Subscript.base_df
        for target_op_name in evaluation_order:
            logging.debug(f"------first pass starting for {target_op_name}")
            target_op_subscript_key = None
            for line in parsed_lines:
                if isinstance(line, ops.Distr):
                    lhs = line.variate
                elif isinstance(line, ops.Assignment):
                    lhs = line.lhs
                    assigned_parameter_keys.add(lhs.get_key())
                    if lhs.get_key() == target_op_name:
                        if lhs.subscript:
                            if lhs.get_key() not in parameter_base_df:
                                raise CompileError(f"Can't resolve {lhs.get_key()}", model_code_string, lhs.line_index,
                                                   lhs.column_index)
                                # raise Exception(f"Can't resolve {lhs.get_key()}")
                            target_op_subscript_key = lhs.subscript.get_key()

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
                            if lhs.subscript is None:
                                raise CompileError(
                                    f"{lhs.get_key()} must declare its subscripts to be used as a reference variable",
                                    model_code_string, lhs.line_index, lhs.column_index)
                                # raise Exception(f"{lhs.get_key()} must declare its subscripts to be used as a reference variable")
                            target_op_subscript_key = lhs.subscript.get_key()
                            line_df = parameter_base_df[lhs.get_key()].copy()
                            line_df.columns = tuple(lhs.subscript.get_key())
                        else:
                            line_df = None
                    else:
                        if not lhs.subscript:
                            break

                # Find all the data referenced in the line
                for subexpr in ops.search_tree(line, ops.Data):
                    data_variables[subexpr.get_key()] = variables.Data(subexpr.get_key(), data_df[subexpr.get_key()])

                # Find all the ways that each parameter is subscripted
                for subexpr in ops.search_tree(line, ops.Param):
                    # Is this handling y = y by ignoring it?
                    if subexpr.get_key() == lhs.get_key():
                        continue
                    subexpr_key = subexpr.get_key()
                    if subexpr.subscript:
                        try:
                            v_df = line_df.loc[:, subexpr.subscript.get_key()]
                        except KeyError as e:
                            raise CompileError(
                                f"Dataframe for {lhs.get_key()} cannot be subscripted by {subexpr.subscript.get_key()}",
                                model_code_string, lhs.line_index, lhs.column_index) from e
                            # raise Exception(f"Dataframe for {lhs.get_key()} cannot be subscripted by {subexpr.subscript.get_key()}") from e
                        logging.debug(f"{subexpr_key} referenced")
                        if subexpr_key in parameter_base_df:
                            v_df.columns = tuple(parameter_base_df[subexpr_key].columns)
                            parameter_base_df[subexpr_key] = (
                                pandas.concat([parameter_base_df[subexpr_key], v_df]).drop_duplicates().reset_index(
                                    drop=True)
                            )
                            for n, key in enumerate(subexpr.subscript.get_key()):
                                parameter_subscript_columns[subexpr_key][n].add(key)
                        else:
                            # initialize the subscript column list length to be the subscript count
                            parameter_subscript_columns[subexpr_key] = [set() for _ in
                                                                        range(len(subexpr.subscript.get_key()))]
                            for n, key in enumerate(subexpr.subscript.get_key()):
                                parameter_subscript_columns[subexpr_key][n].add(key)
                            parameter_base_df[subexpr_key] = v_df.drop_duplicates().reset_index(drop=True)

    def compile(self) -> Tuple[Dict[str, variables.Data], Dict[str, variables.Param], Dict[str, variables.AssignedParam], Dict[str, variables.Subscript], List[LineFunction]]:
        # compiles the expression tree into function statements
        dependency_graph: Dict[str, Set[str]] = generate_dependency_graph(self.expr_tree_list)
        """the dependency graph stores for a key variable x values as variables that we need to know to evaluate.
            If we think of it as rhs | lhs, the dependency graph is an *acyclic* graph that has directed edges going 
            from rhs -> lhs. This is because subscripts on rhs can be resolved, given the dataframe of the lhs. However, for
            assignments, the order is reversed(lhs -> rhs) since we need rhs to infer its subscripts"""

        # traverse through all lines, assuming they are DAGs. This throws operational semantics out the window, but since
        # rat doesn't allow control flow(yet), I'm certain a cycle would mean that it's a mis-specified program
        evaluation_order = topological_sort(dependency_graph)

        logging.debug(f"reverse dependency graph: {dependency_graph}")
        logging.debug(f"first pass eval order: {evaluation_order}")

        self._first_pass()




def compile(data_df: pandas.DataFrame, parsed_lines: List[ops.Expr], model_code_string: str = ""):
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

    # this is the dataframe that goes into variables.Subscript. Each row is a unique combination of indexes
    parameter_base_df: Dict[str, pandas.DataFrame] = {}

    # this is the dictionary that keeps track of what columns the parameter subscripts were defined as
    parameter_subscript_columns: Dict[str, List[Set[str]]] = {}

    # this set keep track of variable names that are not parameters, i.e. assigned by assignment
    assigned_parameter_keys: Set[str] = set()

    data_variables: Dict[str, variables.Data] = {}
    parameter_variables: Dict[str, variables.Param] = {}
    assigned_parameter_variables: Dict[str, variables.AssignedParam] = {}
    variable_subscripts: Dict[str, variables.Subscript] = {}

    line_functions: List[LineFunction] = []

    # first pass : fill param_base_dfs of all parameters
    for target_op_name in evaluation_order:
        logging.debug(f"------first pass starting for {target_op_name}")
        target_op_subscript_key = None
        for line in parsed_lines:
            if isinstance(line, ops.Distr):
                lhs = line.variate
            elif isinstance(line, ops.Assignment):
                lhs = line.lhs
                assigned_parameter_keys.add(lhs.get_key())
                if lhs.get_key() == target_op_name:
                    if lhs.subscript:
                        if lhs.get_key() not in parameter_base_df:
                            raise CompileError(f"Can't resolve {lhs.get_key()}", model_code_string, lhs.line_index, lhs.column_index)
                            #raise Exception(f"Can't resolve {lhs.get_key()}")
                        target_op_subscript_key = lhs.subscript.get_key()

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
                        if lhs.subscript is None:
                            raise CompileError(f"{lhs.get_key()} must declare its subscripts to be used as a reference variable", model_code_string, lhs.line_index, lhs.column_index)
                            #raise Exception(f"{lhs.get_key()} must declare its subscripts to be used as a reference variable")
                        target_op_subscript_key = lhs.subscript.get_key()
                        line_df = parameter_base_df[lhs.get_key()].copy()
                        line_df.columns = tuple(lhs.subscript.get_key())
                    else:
                        line_df = None
                else:
                    if not lhs.subscript:
                        break

            # Find all the data referenced in the line
            for subexpr in ops.search_tree(line, ops.Data):
                data_variables[subexpr.get_key()] = variables.Data(subexpr.get_key(), data_df[subexpr.get_key()])

            # Find all the ways that each parameter is subscripted
            for subexpr in ops.search_tree(line, ops.Param):
                # Is this handling y = y by ignoring it?
                if subexpr.get_key() == lhs.get_key():
                    continue
                subexpr_key = subexpr.get_key()
                if subexpr.subscript:
                    try:
                        v_df = line_df.loc[:, subexpr.subscript.get_key()]
                    except KeyError as e:
                        raise CompileError(f"Dataframe for {lhs.get_key()} cannot be subscripted by {subexpr.subscript.get_key()}", model_code_string, lhs.line_index, lhs.column_index) from e
                        #raise Exception(f"Dataframe for {lhs.get_key()} cannot be subscripted by {subexpr.subscript.get_key()}") from e
                    logging.debug(f"{subexpr_key} referenced")
                    if subexpr_key in parameter_base_df:
                        v_df.columns = tuple(parameter_base_df[subexpr_key].columns)
                        parameter_base_df[subexpr_key] = (
                            pandas.concat([parameter_base_df[subexpr_key], v_df]).drop_duplicates().reset_index(drop=True)
                        )
                        for n, key in enumerate(subexpr.subscript.get_key()):
                            parameter_subscript_columns[subexpr_key][n].add(key)
                    else:
                        # initialize the subscript column list length to be the subscript count
                        parameter_subscript_columns[subexpr_key] = [set() for _ in range(len(subexpr.subscript.get_key()))]
                        for n, key in enumerate(subexpr.subscript.get_key()):
                            parameter_subscript_columns[subexpr_key][n].add(key)
                        parameter_base_df[subexpr_key] = v_df.drop_duplicates().reset_index(drop=True)

        # else:
        #     # Ignore variables in the input dataframe
        #     if target_op_name not in data_df.columns:
        #         raise Exception(
        #             f"Could not find a definition for {target_op_name} (it should either have a prior if it's a parameter or be assigned if it's a transformed parameter)"
        #         )
        if target_op_name in parameter_base_df:
            parameter_base_df[target_op_name].columns = tuple(target_op_subscript_key)

    logging.debug("first pass finished")
    logging.debug("now printing parameter_base_df")
    for key, val in parameter_base_df.items():
        logging.debug(key)
        logging.debug(val)
        variable_subscripts[key] = variables.Subscript(parameter_base_df[key], parameter_subscript_columns[key])
    logging.debug("done printing parameter_base_df")
    logging.debug("now printing variable_subscripts")
    for key, val in variable_subscripts.items():
        logging.debug(key)
        logging.debug(val.base_df)
        logging.info(f"Subscript data for parameter {key}:")
        val.log_summary(logging.INFO)
        logging.info("-----")
    logging.debug("done printing variable_subscripts")

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
            subscript_use_vars: Dict[Tuple, variables.SubscriptUse] = {}

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
                if lhs.subscript is not None:
                    lhs_df = variable_subscripts[lhs.get_key()].df  # parameter_base_df[lhs.get_key()]

            if isinstance(line, ops.Distr):
                if isinstance(lhs, ops.Data):
                    data_vars_used.add(lhs_key)
                    lhs.variable = data_variables[lhs.get_key()]
                elif isinstance(lhs, ops.Param):
                    lower = lhs.lower
                    upper = lhs.upper

                    if lhs_key in variable_subscripts:
                        subexpr = variables.Param(lhs_key, variable_subscripts[lhs_key])
                    else:
                        subexpr = variables.Param(lhs_key)
                    subexpr.set_constraints(float(eval(lower.code())), float(eval(upper.code())))
                    parameter_variables[lhs_key] = subexpr
                    lhs.variable = parameter_variables[lhs_key]

            elif isinstance(line, ops.Assignment):
                assert isinstance(line.lhs, ops.Param), "lhs of assignment must be an Identifier denoting a variable name"
                # assignment lhs are set as variables.AssignedParam, since they're not subject for sampling

                if lhs.subscript:
                    subexpr = variables.AssignedParam(lhs, line.rhs, variable_subscripts[lhs_key])
                    variable_subscripts[lhs.get_key()].incorporate_shifts(lhs.subscript.shifts)
                    subexpr.ops_param.subscript.variable = variables.SubscriptUse(
                        lhs.subscript.get_key(), parameter_base_df[lhs.get_key()], variable_subscripts[lhs_key]
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

                if op.subscript:
                    variable_subscripts[op_key].incorporate_shifts(op.subscript.shifts)

                    index_use_identifier = (op.subscript.get_key(), op.subscript.shifts)

                    df_index = (
                        op.subscript.get_key() if isinstance(lhs, ops.Data) else lhs.variable.subscript.check_and_return_subscripts(op.subscript.get_key())
                    )

                    # Only need to define this particular subscript use once per line (multiple uses will share)
                    if index_use_identifier not in subscript_use_vars:
                        index_use = variables.SubscriptUse(
                            op.subscript.get_key(), lhs_df.loc[:, df_index], variable_subscripts[op_key], op.subscript.shifts
                        )

                        subscript_use_vars[index_use_identifier] = index_use

                    op.subscript.variable = subscript_use_vars[index_use_identifier]

            for op in ops.search_tree(line, ops.Data):
                op_key = op.get_key()
                if op.subscript:
                    raise CompileError(f"Indexing on data variables ({op_key}) not supported", model_code_string, op.line_index, op.column_index)
                    #raise Exception(f"Indexing on data variables ({op_key}) not supported")

            if isinstance(line, ops.Distr):
                line_function = LineFunction(
                    [data_variables[name] for name in data_vars_used],
                    [parameter_variables[name] for name in parameter_vars_used],
                    [assigned_parameter_variables[name] for name in assigned_parameter_vars_used],
                    subscript_use_vars.values(),
                    line,
                )
            elif isinstance(line, ops.Assignment):
                line_function = AssignLineFunction(
                    lhs_key,
                    [data_variables[name] for name in data_vars_used],
                    [parameter_variables[name] for name in parameter_vars_used],
                    [assigned_parameter_variables[name] for name in assigned_parameter_vars_used],
                    subscript_use_vars.values(),
                    line,
                )
            line_functions.append(line_function)
            logging.debug("-------------------")
            break

    logging.debug("2nd pass finished")
    logging.debug(f"data: {list(data_variables.keys())}")
    logging.debug(f"parameters: {list(parameter_variables.keys())}")
    logging.debug(f"assigned parameters {list(assigned_parameter_variables.keys())}")
    return data_variables, parameter_variables, assigned_parameter_variables, variable_subscripts, line_functions
