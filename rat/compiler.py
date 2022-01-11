import itertools
import logging
import numpy
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import Iterable, List, Dict, Set, Tuple, Union
import warnings

from . import ops
from . import variables


class CompileError(Exception):
    def __init__(self, message, code_string: str, line_num: int = -1, column_num: int = -1):
        code_string = code_string.split("\n")[line_num] if code_string else ""
        exception_message = f"An error occured while compiling the following line({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        super().__init__(exception_message)


class Compiler:
    data_df: pandas.DataFrame
    expr_tree_list: List[ops.Expr]
    model_code_string: str

    def __init__(self, data_df: pandas.DataFrame, expr_tree_list: List[ops.Expr], model_code_string: str = ""):
        """
        The rat compiler receives the data dataframe and parsed expression trees and generates Python code which is used
        to declare the target logpdf for inference. During compilation, a total of 2 passes are done through the
        expression tree, excluding 2 separate topological sort operations for identifying compilation order.
        The first pass is based on reverse dependency order and resolves subscripts, while the second pass runs in
        standard dependence order and does the actual codegen/linking with ops.

        In a high level, Compiler does the following operations:

        1. generate reverse dependency graph
        2-1. generate base_df(reference df) for every subscripted parameter
        2-2. generate variable.Subscript() for each variable/data
        3. generate canonical dependency graph
        4-1. generate variable.Param/AssignedParam/Data and variable.SubscriptUse()
        4-2. generate python source for model
        """
        self.data_df = data_df
        self.expr_tree_list = expr_tree_list
        self.model_code_string = model_code_string

        # this is the dataframe that goes into variables.Subscript. Each row is a unique combination of indexes
        self.parameter_base_df: Dict[str, pandas.DataFrame] = {}

        # this is the dictionary that keeps track of what names the parameter subscripts were defined as
        self.parameter_subscript_names: Dict[str, List[Set[str]]] = {}

        # this set keep track of variable names that are not parameters, i.e. assigned by assignment
        self.assigned_parameter_keys: Set[str] = set()

        self.data_variables: Dict[str, variables.Data] = {}
        self.parameter_variables: Dict[str, variables.Param] = {}
        self.assigned_parameter_variables: Dict[str, variables.AssignedParam] = {}
        self.subscript_use_variables: Dict[str, variables.SubscriptUse] = {}
        self.variable_subscripts: Dict[str, variables.Subscript] = {}

    def _generate_dependency_graph(self, reversed=False) -> Dict[str, Set[str]]:
        """
        Generate a dependency graph of all variables/data within self.expr_tree_list.
        Parameter 'reversed', if True, will reverse the standard dependence order of LHS | RHS into RHS | LHS
        """
        dependency_graph: Dict[str, Set[str]] = {}  # this is the dependency graph
        for line in self.expr_tree_list:  # iterate over every line
            if isinstance(line, ops.Distr):
                lhs = line.variate
            elif isinstance(line, ops.Assignment):
                lhs = line.lhs

            lhs_op_key = lhs.get_key()
            if lhs_op_key not in dependency_graph:
                dependency_graph[lhs_op_key] = set()

            for subexpr in ops.search_tree(line, ops.Param, ops.Data):
                rhs_op_key = subexpr.get_key()
                if rhs_op_key == lhs_op_key:
                    continue

                if rhs_op_key not in dependency_graph:
                    dependency_graph[rhs_op_key] = set()

                if reversed:
                    dependency_graph[rhs_op_key].add(lhs_op_key)
                else:
                    dependency_graph[lhs_op_key].add(rhs_op_key)

        return dependency_graph

    def _order_expr_tree(self, dependency_graph: Dict[str, Set[str]]):
        """
        Given a dependence graph, reorder the expression trees in self.expr_tree_list in topological order.
        """
        var_evalulation_order: List[str] = []  # topologically sorted variable names
        current_recursion_lhs = None

        def recursive_order_search(current):
            if current in var_evalulation_order:
                return
            for child in dependency_graph[current]:
                recursive_order_search(child)
            var_evalulation_order.append(current)

        try:
            for expr in self.expr_tree_list:
                if isinstance(expr, ops.Distr):
                    lhs = expr.variate
                elif isinstance(expr, ops.Assignment):
                    lhs = expr.lhs
                current_recursion_lhs = lhs
                recursive_order_search(lhs.get_key())
        except RecursionError as e:
            raise CompileError(
                "Could not topologically order the expression tree. This is because your model is cyclic.",
                self.model_code_string,
                current_recursion_lhs.line_index,
                current_recursion_lhs.column_index,
            ) from e

        # create (Expr, lhs_var_name) pair, which is used to sort based on lhs_var_name
        expr_tree_list = [[x, x.lhs.get_key() if isinstance(x, ops.Assignment) else x.variate.get_key()] for x in self.expr_tree_list]

        def get_index(name, expr):
            try:
                return var_evalulation_order.index(name)
            except ValueError as e:
                raise CompileError(
                    f"First pass error: key {name} not found while ordering expressions. Please check that all variables/parameters have been declared.",
                    self.model_code_string,
                    line_num=expr.line_index,
                    column_num=0,
                ) from e

        return [elem[0] for elem in sorted(expr_tree_list, key=lambda x: get_index(x[1], x[0]))]

    def _build_base_df(self, rev_ordered_expr_trees: List[ops.Expr]):
        """
        First code pass. This function must receive the *reversed* dependence graph, or RHS | LHS where edges go from
        RHS to LHS. This function will loop over the expression trees once, and write the parameter base dataframes into
        the class attribute self. parameter_base_df.

        The core two classes related to subscripts are `variables.Subscript` and `variables.SubscriptUse`. In short,
        `Subscript` denotes the dataframe where the variable is used in LHS. It's used as the basis for resolving
        subscripts on its RHS.
        Meanwhile, `SubscriptUse` is what's given to Param in RHS. It's inferred from the LHS variable's `Subscript`,
        which is indexed according to the RHS subscript.

        1. generate base_df for every LHS variable
        2. identify every subscript scheme for all subscripted variables
        """
        # On the first pass, we generate the base reference dataframe for each variable. This goes into
        # variable.Subscript.base_df

        for top_expr in rev_ordered_expr_trees:
            if isinstance(top_expr, ops.Distr):
                lhs: ops.Expr = top_expr.variate
            elif isinstance(top_expr, ops.Assignment):
                lhs: ops.Expr = top_expr.lhs

                # check if lhs type is ops.Param (can't assign to data)
                if not isinstance(lhs, ops.Param):
                    raise CompileError(
                        "LHS of assign statement must be a variable and not data!", self.model_code_string, lhs.line_index, lhs.column_index
                    )

            else:
                # This should never be reached given a well-formed expression tree
                raise CompileError(
                    f"Top-level expressions may only be an assignment or sample statement, but got type {top_expr.__class__.__name__}",
                    self.model_code_string,
                    top_expr.line_index,
                    top_expr.column_index,
                )

            lhs_key = lhs.get_key()

            # we now construct the LHS reference dataframe. This is the dataframe that the LHS variable possesses.
            # For example, consider `slope[age, country] ~ normal(mu[country], 1);`
            # The subscript "country" on the RHS is looked up from the LHS dataframe's "country" column.
            # As you can see, for every line that contains subscripted variables, the LHS's subscripts must be known
            # in order to resolve the correct subscript values on the RHS. lhs_base_df is the reference base
            # dataframe that the RHS subscripts are resolved on that line.
            if isinstance(lhs, ops.Data):
                # If the LHS type is data, the reference dataframe is the data dataframe
                lhs_base_df = self.data_df

            elif isinstance(lhs, ops.Param):
                # Otherwise, we take the reference dataframe of the LHS variable.
                # Since the lines are evaluated in a topological order, the LHS would have been known at this point.
                if lhs.subscript:
                    # subscripts for variables come from the LHS of the line that the variable was used
                    if lhs.get_key() in self.parameter_base_df:
                        lhs_base_df = self.parameter_base_df[lhs.get_key()].copy()
                        # rename the column names(subscript names) of the base_df
                        # to be compatible on the current line
                        try:
                            lhs_base_df.columns = tuple(lhs.subscript.get_key())
                        except ValueError as e:
                            # if usage subscript length and declared subscript length do not match, raise exception
                            raise CompileError(
                                f"Variable {lhs.get_key()}, has declared {len(lhs.subscript.get_key())}LHS subscript, but was used in RHS as {len(lhs_base_df.columns)} subscripts.",
                                self.model_code_string,
                                lhs.line_index,
                                lhs.column_index,
                            ) from e
                    else:
                        raise CompileError(
                            f"Variable {lhs.get_key()} was declared to have subscript {lhs.subscript.get_key()}, but no usage of said variable was found.",
                            self.model_code_string,
                            lhs.line_index,
                            lhs.column_index,
                        )
                else:
                    lhs_base_df = None

            for sub_expr in ops.search_tree(top_expr, ops.Param):
                sub_expr_key = sub_expr.get_key()
                # we're only interested in the RHS usages:
                if sub_expr.get_key() == lhs.get_key():
                    continue

                # if the variable is not subscripted, nothing more needs to be done
                if not sub_expr.subscript:
                    continue

                # get the subscripted columns from LHS base df
                try:
                    rhs_base_df = lhs_base_df.loc[:, sub_expr.subscript.get_key()]
                except KeyError as e:
                    # this means a subscript referenced on RHS does not exist on the LHS base dataframe
                    raise CompileError(
                        f"Couldn't index RHS variable {sub_expr_key}'s declared subscript{sub_expr.subscript.get_key()} from LHS base df, which has subscripts {tuple(lhs_base_df.columns)}.",
                        self.model_code_string,
                        sub_expr.line_index,
                        sub_expr.column_index,
                    ) from e
                except AttributeError as e:
                    # this means LHS has no base dataframe but a subscript was attempted from RHS
                    raise CompileError(
                        f"Variable on RHS side '{sub_expr_key}' attempted to subscript {sub_expr.subscript.get_key()}, but LHS variable '{lhs.get_key()}' has no subscripts!",
                        self.model_code_string,
                        sub_expr.line_index,
                        sub_expr.column_index,
                    ) from e

                if sub_expr_key in self.parameter_base_df:
                    # if we already have a base_df registered, try merging them
                    try:
                        rhs_base_df.columns = tuple(self.parameter_base_df[sub_expr_key])
                    except ValueError as e:
                        raise CompileError(
                            f"Subscript length mismatch: variable on RHS side '{sub_expr_key}' has declared subscript of length {len(sub_expr_key)}, but has been used with a subscript of length {len(self.parameter_base_df[sub_expr_key])}",
                            self.model_code_string,
                            sub_expr.line_index,
                            sub_expr.column_index,
                        ) from e
                    self.parameter_base_df[sub_expr_key] = (
                        pandas.concat([self.parameter_base_df[sub_expr_key], rhs_base_df]).drop_duplicates().reset_index(drop=True)
                    )

                    # keep track of subscript names for each position
                    for n, subscript in enumerate(sub_expr.subscript.get_key()):
                        self.parameter_subscript_names[sub_expr_key][n].add(subscript)

                else:  # or generate base df for RHS variable if it doesn't exist
                    # this list of sets keep track of the aliases of each subscript position
                    self.parameter_subscript_names[sub_expr_key] = [set() for _ in range((len(sub_expr.subscript.get_key())))]
                    for n, subscript in enumerate(sub_expr.subscript.get_key()):
                        self.parameter_subscript_names[sub_expr_key][n].add(subscript)

                    self.parameter_base_df[sub_expr_key] = rhs_base_df.drop_duplicates().reset_index(drop=True)

            if lhs_key in self.parameter_base_df:
                # Update base_df's column names to the latest subscript definitions.
                # This means the index names will be set to the subscript names of the LHS definition
                self.parameter_base_df[lhs_key].columns = tuple(lhs.subscript.get_key())

    def _build_variables(self, ordered_expr_trees: List[ops.Expr]):
        """
        The goal in the second pass is
        1. Generate variable.Param, variable.AssignedParam, variable.Data for each respective variable/data
        2. Generate variable.SubscriptUse and map them to variable.Param/AssignedParam. They are used to resolve RHS
        subscripts from the LHS base_df

        The expression trees must be sorted in canonical topological order(LHS | RHS) in order to run succesfully.
        """

        def unique_subscript_use_itentifier_generator():
            identifier = 0

            while True:
                yield repr(identifier)
                identifier += 1

        unique_subscript_use_itentifier = unique_subscript_use_itentifier_generator()

        subscript_use_vars: List[variables.SubscriptUse] = []
        for top_expr in ordered_expr_trees:
            logging.debug("@" * 5)

            if isinstance(top_expr, ops.Distr):
                lhs = top_expr.variate
            elif isinstance(top_expr, ops.Assignment):
                lhs = top_expr.lhs
            else:
                raise CompileError(
                    f"Top-level expressions may only be an assignment or sample statement, but got type {top_expr.__class__.__name__}",
                    self.model_code_string,
                    top_expr.line_index,
                    top_expr.column_index,
                )

            lhs_key = lhs.get_key()
            logging.debug(f"Second pass for {lhs_key}")
            if isinstance(lhs, ops.Data):
                lhs_base_df = self.data_df
            elif isinstance(lhs, ops.Param):
                if lhs.subscript is not None:
                    lhs_base_df = self.parameter_base_df[lhs_key]
                else:
                    lhs_base_df = None
            else:
                raise CompileError(
                    f"LHS of statement should be a Data or Param type, but got {lhs.__class__.__name__}",
                    self.model_code_string,
                    lhs.line_index,
                    lhs.column_index,
                )

            if isinstance(top_expr, ops.Distr):
                # If we're working with a sampling statement, create a variable object and link them to ops.Expr
                if isinstance(lhs, ops.Data):
                    self.data_variables[lhs_key] = variables.Data(lhs_key, self.data_df[lhs_key])
                    lhs.variable = self.data_variables[lhs_key]

                elif isinstance(lhs, ops.Param):
                    try:  # evaluate constraint expressions
                        lower_constraint_value = float(eval(lhs.lower.code()))
                        upper_constraint_value = float(eval(lhs.upper.code()))
                    except Exception as e:
                        raise CompileError(
                            f"Error when evaluating constraints for {lhs_key}. Constraint expressions must be able to be evaluated at compile time.",
                            self.model_code_string,
                            lhs.line_index,
                            lhs.column_index,
                        ) from e

                    # generate variables.Param for LHS. This is because the variable on LHS is first seen here.
                    if lhs.subscript is not None:
                        lhs_variable = variables.Param(lhs_key, self.variable_subscripts[lhs_key])
                    else:
                        lhs_variable = variables.Param(lhs_key)

                    # apply constraints
                    lhs_variable.set_constraints(lower_constraint_value, upper_constraint_value)
                    lhs.variable = lhs_variable  # bind to ops.Param in expression tree

                    self.parameter_variables[lhs_key] = lhs_variable

                else:
                    # should be unreachable since parser checks types
                    raise CompileError(
                        f"LHS for sample statement has invalid type", self.model_code_string, lhs.line_index, lhs.column_index
                    )
            elif isinstance(top_expr, ops.Assignment):
                # Same procedure with assignment statements: create variable object and link back to ops.Expr
                if not isinstance(lhs, ops.Param):  # shouldn't be reachable because parser catches type errors
                    raise CompileError(
                        "LHS for assignment statement has invalid type", self.model_code_string, lhs.line_index, lhs.column_index
                    )

                # generate variables.AssignedParam for LHS. This is because the variable on LHS is first seen here.
                if lhs.subscript is not None:
                    lhs_variable = variables.AssignedParam(lhs, top_expr.rhs, self.variable_subscripts[lhs_key])
                else:
                    lhs_variable = variables.AssignedParam(lhs, top_expr.rhs)

                self.assigned_parameter_variables[lhs_key] = lhs_variable
                lhs.variable = lhs_variable  # link variable.Data to op.Data in expression tree

            #
            # We now finished creating variable types for the LHS, which is the first time the variable was seen.
            # Now we resolve the subscripts for each variable type on RHS, by creating SubscriptUse and linking them
            # into each variables.Param in the line.
            for subexpr in ops.search_tree(top_expr, ops.Data):
                subexpr_key = subexpr.get_key()
                if subexpr_key == lhs_key:
                    # We don't manipulate the LHS variable that we just created
                    continue
                self.data_variables[subexpr_key] = variables.Data(subexpr_key, self.data_df[subexpr_key])
                subexpr.variable = self.data_variables[subexpr_key]

            for subexpr in ops.search_tree(top_expr, ops.Param):
                # iterate through parameters/assigned parameters
                subexpr_key = subexpr.get_key()

                # add parameter name to referenced data set and link to ops
                if subexpr_key in self.assigned_parameter_variables:
                    # link variable.AssignedParam to op.Param
                    subexpr.variable = self.assigned_parameter_variables[subexpr_key]
                elif subexpr_key in self.parameter_variables:
                    # link variable.Param to op.Param
                    subexpr.variable = self.parameter_variables[subexpr_key]

                else:
                    # if a param does not exist yet, this means either:
                    # 1. The model's computational graph is not a DAG
                    # 2. The parameter is undefined
                    raise CompileError(
                        f"Could not find a prior declared for variable {subexpr_key}.",
                        self.model_code_string,
                        subexpr.line_index,
                        subexpr.column_index,
                    )

                # resolve subscripts
                if subexpr.subscript is not None:
                    # incorporate shifts into variable.Subscript
                    self.variable_subscripts[subexpr_key].incorporate_shifts(subexpr.subscript.shifts)

                    # get the base df's reference column names
                    base_df_index = (
                        subexpr.subscript.get_key()
                        if isinstance(lhs, ops.Data)
                        else lhs.variable.subscript.check_and_return_subscripts(subexpr.subscript.get_key())
                    )

                    # extract the subscripts from the base df and link them into variable.SubscriptUse
                    variable_subscript_use = variables.SubscriptUse(
                        subexpr.subscript.get_key(),
                        lhs_base_df.loc[:, base_df_index],
                        self.variable_subscripts[subexpr_key],
                        next(unique_subscript_use_itentifier),
                        subexpr.subscript.shifts,
                    )

                    # link the created variable.SubscriptUse into ops.Param
                    subexpr.subscript.variable = variable_subscript_use
                    self.subscript_use_variables[variable_subscript_use.code()] = variable_subscript_use

            # quickly check to make sure subscripts don't exist for Data
            for subexpr in ops.search_tree(top_expr, ops.Data):
                subexpr_key = subexpr.get_key()
                if subexpr.subscript:
                    raise CompileError(
                        f"Indexing on data variables ({subexpr_key}) not supported",
                        self.model_code_string,
                        subexpr.line_index,
                        subexpr.column_index,
                    )

            # self.line_functions.append(line_function)
            logging.debug(f"data: {list(self.data_variables.keys())}")
            logging.debug(f"parameters: {list(self.parameter_variables.keys())}")
            logging.debug(f"assigned parameters {list(self.assigned_parameter_variables.keys())}")

    def _generate_python_source(self, ordered_expr_trees: List[ops.Expr]):
        """
        Return python source containing functions for constraining and transforming variables and
        evaluating all the log densities
        """
        # Get parameter dimensions so can build individual arguments from
        # unconstrained vector
        unconstrained_parameter_size = 0
        parameter_names = []
        parameter_offsets = []
        parameter_sizes = []
        for name, parameter in self.parameter_variables.items():
            parameter_names.append(name)
            parameter_offsets.append(unconstrained_parameter_size)
            parameter_size = parameter.size()
            parameter_sizes.append(parameter_size)

            if parameter_size is not None:
                unconstrained_parameter_size += parameter_size
            else:
                unconstrained_parameter_size += 1

        # Generate code for unconstraining and transforming parameters
        code = (
            f"import rat.constraints\n"
            f"import rat.math\n"
            f"import jax\n"
            f"\nunconstrained_parameter_size = {unconstrained_parameter_size}\n"
            f"\ndef constrain_parameters(unconstrained_parameter_vector, pad=True):\n"
            f"    unconstrained_parameters = {{}}\n"
            f"    parameters = {{}}\n"
            f"    total = 0.0\n"
        )

        # Constrain parameters
        for name, offset, size in zip(parameter_names, parameter_offsets, parameter_sizes):
            unconstrained_reference = f"unconstrained_parameters['{name}']"
            constrained_reference = f"parameters['{name}']"
            code += f"\n    # {name}\n"
            if size is not None:
                code += f"    {unconstrained_reference} = unconstrained_parameter_vector[..., {offset} : {offset + size}]\n"
            else:
                code += f"    {unconstrained_reference} = unconstrained_parameter_vector[..., {offset}]\n"

            variable = self.parameter_variables[name]
            lower = variable.lower
            upper = variable.upper

            if lower > float("-inf") or upper < float("inf"):
                if lower > float("-inf") and upper == float("inf"):
                    code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.lower({unconstrained_reference}, {lower})\n"
                elif lower == float("inf") and upper < float("inf"):
                    code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.upper({unconstrained_reference}, {upper})\n"
                elif lower > float("-inf") and upper < float("inf"):
                    code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.finite({unconstrained_reference}, {lower}, {upper})\n"

                code += "    total += jax.numpy.sum(constraints_jacobian_adjustment)\n"
            else:
                code += f"    {constrained_reference} = {unconstrained_reference}\n"

            if size is not None and size != variable.padded_size():
                code += f"    if pad:\n"
                code += f"        {constrained_reference} = jax.numpy.pad({constrained_reference}, (0, {variable.padded_size() - size}))\n"
        code += "\n    return total, parameters\n"

        # Transform parameters
        code += "\ndef transform_parameters(data, subscripts, parameters):"
        for top_expr in ordered_expr_trees:
            if isinstance(top_expr, ops.Assignment):
                lhs_key = top_expr.lhs.get_key()
                code += f"\n    # {lhs_key}\n"
                code += f"    parameters['{lhs_key}'] = {top_expr.rhs.code()}\n"
        code += "\n    return parameters\n"

        # Generate code for evaluating densities
        code += f"\ndef evaluate_densities(data, subscripts, parameters):\n" f"    target = 0.0\n"

        for top_expr in ordered_expr_trees:
            if isinstance(top_expr, ops.Distr):
                code += f"    target += jax.numpy.sum({top_expr.code()})\n"
        code += "\n    return target"

        return code

    def compile(
        self,
    ) -> Tuple[
        Dict[str, variables.Data],
        Dict[str, variables.Param],
        Dict[str, variables.AssignedParam],
        Dict[str, variables.SubscriptUse],
        str,
    ]:
        if self.parameter_variables:
            raise Exception("Compiler.compile() may be invoked only once per instance. Create a new instance to recompile.")

        # compiles the expression tree into function statements
        dependency_graph: Dict[str, Set[str]] = self._generate_dependency_graph(reversed=True)
        """the dependency graph stores for a key variable x values as variables that we need to know to evaluate.
            If we think of it as rhs | lhs, the dependency graph is an *acyclic* graph that has directed edges going 
            from rhs -> lhs. This is because subscripts on rhs can be resolved, given the dataframe of the lhs. However, for
            assignments, the order is reversed(lhs -> rhs) since we need rhs to infer its subscripts"""

        # traverse through all lines, assuming they are DAGs.
        evaluation_order: List[ops.Expr] = self._order_expr_tree(dependency_graph)

        logging.debug(f"reverse dependency graph: {dependency_graph}")
        logging.debug(f"first pass line eval order: {[x.line_index for x in evaluation_order]}")

        # build the parameter_base_df for each variable
        self._build_base_df(evaluation_order)

        # build variable.Subscript for each parameter
        logging.debug("now printing parameter_base_df")
        for var_name, subscript in self.parameter_subscript_names.items():
            logging.debug(var_name)
            logging.debug(self.parameter_base_df[var_name])
            self.variable_subscripts[var_name] = variables.Subscript(
                self.parameter_base_df[var_name], self.parameter_subscript_names[var_name]
            )
        logging.debug("done printing parameter_base_df")
        logging.debug("now printing variable_subscripts")

        for key, val in self.variable_subscripts.items():
            logging.debug(key)
            logging.debug(val.base_df)
            logging.info(f"Subscript data for parameter {key}:")
            val.log_summary(logging.INFO)
            logging.info("-----")
        logging.debug("done printing variable_subscripts")

        # rebuild the dependency graph in normal order, that is lhs | rhs. We now convert the expression trees into
        # Python code
        dependency_graph = self._generate_dependency_graph()

        # and reorder the expression trees accordingly
        evaluation_order = self._order_expr_tree(dependency_graph)

        logging.debug(f"reverse dependency graph: {dependency_graph}")
        logging.debug(f"second pass line eval order: {[x.line_index for x in evaluation_order]}")

        # run seconds pass, which binds subscripts and does codegen
        self._build_variables(evaluation_order)

        return (
            self.data_variables,
            self.parameter_variables,
            self.assigned_parameter_variables,
            self.subscript_use_variables,
            self._generate_python_source(evaluation_order),
        )
