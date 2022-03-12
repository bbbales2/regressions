import collections
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
    # There are a couple places internally here where it is handy to have a unique
    # identifier. We can use increasing integers to do this
    _unique_identifier: int

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
        self._unique_identifier = 0

        # these dataframes that go into variables.Subscript. Each row is a unique combination of indexes
        self.parameter_base_df: Dict[str, pandas.DataFrame] = {}

        # these dataframes that go into variables.Subscript
        self.data_base_df: Dict[str, pandas.DataFrame] = {}

        # this is the dictionary that keeps track of what names the parameter subscripts were defined as
        self.parameter_subscript_names: Dict[str, List[Set[str]]] = {}

        # this set keep track of variable names that are not parameters, i.e. assigned by assignment
        self.assigned_parameter_keys: Set[str] = set()

        self.data_variables: Dict[str, variables.Data] = {}
        self.parameter_variables: Dict[str, variables.Param] = {}
        self.assigned_parameter_variables: Dict[str, variables.AssignedParam] = {}
        self.subscript_use_variables: Dict[str, variables.SubscriptUse] = {}
        self.variable_subscripts: Dict[str, variables.Subscript] = {}
        self.first_in_group_indicators: Dict[str, numpy.ndarray] = {}

    def _get_unique_identifier(self) -> str:
        """
        Return an as-yet un-returned string. This is useful for generating unique identifiers
        for variables
        """
        self._unique_identifier += 1
        return repr(self._unique_identifier)

    def _identify_primary_symbols(self):
        # figure out which symbols will have dataframes
        has_dataframe = set()
        for top_expr in self.expr_tree_list:
            for primeable_symbol in ops.search_tree(top_expr, ops.PrimeableExpr):
                if isinstance(primeable_symbol, ops.Data) or primeable_symbol.subscript is not None:
                    has_dataframe.add(primeable_symbol.get_key())

        for top_expr in self.expr_tree_list:
            # We compute the primary variable reference in a line of code with the rules:
            # 1. There can only be one primary variable reference (priming two references to the same variable is still an error)
            # 2. If a variable is marked as primary, then it is the primary variable.
            # 3. If there is no marked primary variable, then all variables with dataframes are treated as prime.
            # 4. If there are no variables with dataframes, the leftmost one is the primary one
            # 5. It is an error if no primary variable can be identified
            primary_symbol: ops.PrimeableExpr = None
            # Rule 2
            for primeable_symbol in ops.search_tree(top_expr, ops.PrimeableExpr):
                if primeable_symbol.prime:
                    if primary_symbol is None:
                        primary_symbol = primeable_symbol
                    else:
                        msg = f"Found two marked primary variables {primary_symbol.get_key()} and {primeable_symbol.get_key()}. There should only be one"
                        raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)

            # Rule 3
            if primary_symbol is None:
                for primeable_symbol in ops.search_tree(top_expr, ops.PrimeableExpr):
                    primeable_key = primeable_symbol.get_key()

                    if primeable_key in has_dataframe:
                        if primary_symbol is None:
                            primary_symbol = primeable_symbol
                        else:
                            primary_key = primary_symbol.get_key()
                            if primary_key != primeable_key:
                                msg = f"No marked primary variable and at least {primary_key} and {primeable_key} are candidates. A primary variable should be marked manually"
                                raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)
                            else:
                                msg = f"No marked primary variable but found multiple references to {primary_key}. One reference should be marked manually"
                                raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)

            # Rule 4
            if primary_symbol is None:
                for primeable_symbol in ops.search_tree(top_expr, ops.PrimeableExpr):
                    primary_symbol = primeable_symbol
                    break

            # Rule 5
            if primary_symbol is None:
                msg = f"No primary variable found on line (this means there are no candidate variables)"
                raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)

            # Mark the primary symbol if it wasn't already
            primary_symbol.prime = True

    def _get_primary_symbol_from_statement(self, top_expr):
        """
        Get the primary symbol in a statement. This assumes that the statement has
        only one primary symbol
        """
        for primary_symbol in ops.search_tree(top_expr, ops.PrimeableExpr):
            if primary_symbol.prime:
                break
        else:
            msg = f"Internal compiler error. No primary variable found"
            raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)
        return primary_symbol

    def _build_base_df(self):
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

        # Make sure that all secondary parameter uses precede primary uses (or throw an error)
        N_lines = len(self.expr_tree_list)
        for i, top_expr in enumerate(self.expr_tree_list):
            primary_symbol = self._get_primary_symbol_from_statement(top_expr)

            primary_key = primary_symbol.get_key()

            # There aren't any restrictions on primary/secondary data
            if isinstance(primary_symbol, ops.Data):
                continue

            for j in range(i + 1, N_lines):
                for secondary_symbol in ops.search_tree(self.expr_tree_list[j], ops.Param):
                    secondary_key = secondary_symbol.get_key()

                    if primary_key == secondary_key and not secondary_symbol.prime:
                        msg = f"Primary variable {primary_symbol.get_key()} used on line {i} but then referenced as non-prime on line {j}. The primed uses must come last"
                        raise CompileError(msg, self.model_code_string, primary_symbol.line_index, primary_symbol.column_index)

        for top_expr in self.expr_tree_list:
            primary_symbol = self._get_primary_symbol_from_statement(top_expr)
            # Identify the primary dataframe if there is one and resolve naming issues
            primary_key = primary_symbol.get_key()

            match primary_symbol:
                case ops.Data():
                    # If the primary primary_symbol is data, the reference dataframe is the data dataframe
                    primary_df = self.data_df
                # TODO: This isn't implemented but might make sense
                # case ops.Subscript():
                # If the primary primary_symbol is a subscript, the reference dataframe is the data dataframe
                #    primary_df = self.data_df
                case ops.Param():
                    # For parameters
                    # 1. A subscripted primary parameter primary_symbol must have a dataframe
                    # 2. If there is no subscript, then:
                    #    A. If a dataframe exists, then column names must be uniquely defined
                    #    B. If no dataframe exists, the parameter is a scalar
                    # Since the lines are evaluated in a topological order, the rows of the primary dataframe are known at this point.
                    if primary_key in self.parameter_base_df:
                        primary_df = self.parameter_base_df[primary_key].copy()

                        # If the primary primary_symbol is subscripted, should be a dataframe
                        if primary_symbol.subscript:
                            # Handle subscript renaming
                            try:
                                primary_df.columns = tuple(primary_symbol.subscript.get_key())
                            except ValueError as e:
                                # if usage subscript length and declared subscript length do not match, raise exception
                                msg = f"Rename {primary_symbol.code()} not possible, {len(primary_df.columns)} required names but {len(primary_symbol.subscript.get_key())} given."
                                raise CompileError(
                                    msg, self.model_code_string, primary_symbol.line_index, primary_symbol.column_index
                                ) from e

                            self.parameter_base_df[primary_key].columns = primary_df.columns
                        else:
                            for n, in_use_subscripts in enumerate(self.parameter_subscript_names[primary_key]):
                                if len(in_use_subscripts) > 1:
                                    msg = f"Subscript(s) of {primary_key} must be renamed, the name for subscript {n} is ambiguous ({in_use_subscripts})"
                                    raise CompileError(msg, self.model_code_string, primary_symbol.line_index, primary_symbol.column_index)
                    else:
                        primary_df = None
                case _:
                    msg = f"Expected a parameter or data primary_symbol but got type {primary_symbol.__class__.__name__}"
                    raise CompileError(msg, self.model_code_string, primary_symbol.line_index, primary_symbol.column_index)

            # For each line, we take the dataframe of the primary variable and append as necessary
            # rows to the dataframes of all the secondary variables
            #
            # Variables are only created if they are used and this will match the semantics of the
            # match statements
            dfs_and_params: List[Tuple[pandas.DataFrame, List[ops.Param]]] = []
            match top_expr:
                # case ops.MatchedStatement():
                #     sorted_primary_df = primary_df.sort_values(top_expr.bounded_variable_name)
                #     first_expr = top_expr.initial_declarations["first"]
                #     recurrence_expr = top_expr.recurrence_equation
                #     lhs_params = list(ops.search_tree(top_expr.lhs, ops.Param))
                #     first_expr_params = list(ops.search_tree(first_expr, ops.Param))
                #     recurrence_expr_params = list(ops.search_tree(recurrence_expr, ops.Param))
                #     dfs_and_params.append((sorted_primary_df.iloc[:1], lhs_params + first_expr_params))
                #     dfs_and_params.append((sorted_primary_df.iloc[1:], lhs_params + recurrence_expr_params))
                case _:
                    expr_params = list(ops.search_tree(top_expr, ops.Param))
                    dfs_and_params.append((primary_df, expr_params))

            for subset_primary_df, params in dfs_and_params:
                # Find every other parameter in the line and attempt to construct data
                # frames for the ones with subscripts
                for param in params:
                    param_key = param.get_key()

                    # no need to check the primary symbol
                    if param_key == primary_key:
                        continue

                    # if the symbol is not subscripted, nothing more needs to be done
                    if not param.subscript:
                        continue

                    # get the subscripted columns from primary df
                    try:
                        use_df = subset_primary_df.loc[:, param.subscript.get_key()]
                    except KeyError as e:
                        # this means a subscript referenced does not exist in the primary dataframe
                        msg = f"Couldn't index RHS variable {param_key}'s declared subscript{param.subscript.get_key()} from LHS base df, which has subscripts {tuple(primary_df.columns)}."
                        raise CompileError(msg, self.model_code_string, param.line_index, param.column_index) from e
                    except AttributeError as e:
                        # this means LHS has no base dataframe but a subscript was attempted from RHS
                        msg = f"Variable {param_key} attempted to subscript using {param.subscript.get_key()}, but primary variable {primary_key} has no subscripts!"
                        raise CompileError(msg, self.model_code_string, param.line_index, param.column_index) from e

                    if param_key in self.parameter_base_df:
                        # if we already have a base_df registered, try merging them
                        try:
                            use_df.columns = tuple(self.parameter_base_df[param_key])
                        except ValueError as e:
                            msg = f"Subscript length mismatch: variable on RHS side {param_key} has declared subscript of length {len(param_key)}, but has been used with a subscript of length {len(self.parameter_base_df[param_key])}"
                            raise CompileError(msg, self.model_code_string, param.line_index, param.column_index) from e

                        self.parameter_base_df[param_key] = (
                            pandas.concat([self.parameter_base_df[param_key], use_df]).drop_duplicates().sort_values(list(use_df.columns)).reset_index(drop=True)
                        )

                        # keep track of subscript names for each position
                        for n, subscript in enumerate(param.subscript.get_key()):
                            self.parameter_subscript_names[param_key][n].add(subscript)

                    else:  # or generate base df for RHS variable if it doesn't exist
                        # this list of sets keep track of the aliases of each subscript position
                        self.parameter_subscript_names[param_key] = [set() for _ in range((len(param.subscript.get_key())))]
                        for n, subscript in enumerate(param.subscript.get_key()):
                            self.parameter_subscript_names[param_key][n].add(subscript)
                        self.parameter_base_df[param_key] = (
                            use_df.drop_duplicates().sort_values(list(use_df.columns)).reset_index(drop=True)
                        )

            # TODO: This check will need to get fancier when there are multiple dataframes
            # Find every Data in the line and make sure that they all have the necessary subscripts
            for data in ops.search_tree(top_expr, ops.Data):
                data_key = data.get_key()

                self.data_base_df[data_key] = self.data_df

    def _build_variables(self):
        """
        The goal in the second pass is
        1. Generate variable.Param, variable.AssignedParam, variable.Data for each respective variable/data
        2. Generate variable.SubscriptUse and map them to variable.Param/AssignedParam. They are used to resolve RHS
        subscripts from the LHS base_df

        The expression trees must be sorted in canonical topological order(LHS | RHS) in order to run succesfully.
        """

        # Parameters cannot be used after they are assigned
        N_lines = len(self.expr_tree_list)
        for i, top_expr in enumerate(self.expr_tree_list):
            match top_expr:
                case ops.Assignment(lhs):
                    for j in range(i + 1, N_lines):
                        lhs_key = lhs.get_key()
                        for symbol in ops.search_tree(self.expr_tree_list[j], ops.Param):
                            symbol_key = symbol.get_key()
                            if symbol_key == lhs_key:
                                msg = f"Parameter {lhs.get_key()} is assigned on line {i} but used on line {j}. A variable cannot be used after it is assigned"
                                raise CompileError(msg, self.model_code_string, lhs.line_index, lhs.column_index)

        ordered_expr_trees = list(reversed(self.expr_tree_list))

        def unique_subscript_use_itentifier_generator():
            identifier = 0

            while True:
                yield repr(identifier)
                identifier += 1

        unique_subscript_use_itentifier = unique_subscript_use_itentifier_generator()

        # Create variable objects
        for top_expr in ordered_expr_trees:
            # if top_expr is an assignment or a match statement, pull out the left hand side and treat that specially
            match top_expr:
                case ops.Assignment(lhs, remainder):
                    lhs_key = lhs.get_key()

                    if not isinstance(lhs, ops.Param):
                        msg = f"Left hand side for assignment statement must be a parameter instead of {lhs.__class__.__name__}"
                        raise CompileError(msg, self.model_code_string, lhs.line_index, lhs.column_index)

                    # create the variable if necessary
                    if lhs_key not in self.assigned_parameter_variables:
                        # if the underlying variable has a dataframe, there will be an entry in variable_subscripts
                        if lhs_key in self.variable_subscripts:
                            lhs_variable = variables.AssignedParam(lhs, remainder, self.variable_subscripts[lhs_key])
                        else:
                            lhs_variable = variables.AssignedParam(lhs, remainder)

                    # if this variable is stored in parameter_variables, remove the entry there
                    if lhs_key in self.parameter_variables:
                        self.parameter_variables.pop(lhs_key)

                    self.assigned_parameter_variables[lhs_key] = lhs_variable
                case remainder:
                    pass

            # remainder at this point is the part of top_expr that is not the left hand side of an assignment
            # or a match statement
            for symbol in ops.search_tree(remainder, ops.Data, ops.Param):
                symbol_key = symbol.get_key()

                match symbol:
                    case ops.Data():
                        # create the variable if necessary
                        if symbol_key not in self.data_variables:
                            self.data_variables[symbol_key] = variables.Data(
                                name=symbol_key,
                                series=self.data_base_df[symbol_key].loc[:, symbol_key],
                                subscript=self.variable_subscripts[symbol_key],
                            )
                    case ops.Param():
                        # create the variable if necessary
                        if symbol_key not in self.assigned_parameter_variables and symbol_key not in self.parameter_variables:
                            # if the underlying variable has a dataframe, there will be an entry in variable_subscripts
                            if symbol_key in self.variable_subscripts:
                                variable = variables.Param(name=symbol_key, subscript=self.variable_subscripts[symbol_key])
                            else:
                                variable = variables.Param(name=symbol_key)

                            self.parameter_variables[symbol_key] = variable
                    case _:
                        msg = f"Internal compiler error"
                        raise CompileError(msg, self.model_code_string, lhs.line_index, lhs.column_index)

        # Handle constraints
        for top_expr in ordered_expr_trees:
            for parameter_symbol in ops.search_tree(top_expr, ops.Param):
                parameter_key = parameter_symbol.get_key()

                try:  # evaluate constraint expressions
                    lower_expr = parameter_symbol.lower
                    upper_expr = parameter_symbol.upper

                    lower_constraint_value = float(eval(lower_expr.code())) if lower_expr is not None else None
                    upper_constraint_value = float(eval(upper_expr.code())) if upper_expr is not None else None
                except Exception as e:
                    msg = f"Error when evaluating constraints for {parameter_key}. Constraint expressions must be able to be evaluated at compile time."
                    raise CompileError(msg, self.model_code_string, parameter_symbol.line_index, parameter_symbol.column_index) from e

                # we only pay attention to constraints on actual parameters
                if parameter_key in self.parameter_variables:
                    parameter_variable = self.parameter_variables[parameter_key]

                    # apply constraints
                    try:
                        parameter_variable.set_constraints(lower_constraint_value, upper_constraint_value)
                    except Exception as e:
                        msg = f"Constraints of {parameter_key} have already been set"
                        raise CompileError(msg, self.model_code_string, parameter_symbol.line_index, parameter_symbol.column_index)

        # Link ops back to variables
        for top_expr in ordered_expr_trees:
            for symbol in ops.search_tree(top_expr, ops.Data, ops.Param):
                symbol_key = symbol.get_key()

                match symbol:
                    case ops.Data():
                        # point symbol to appropriate variable
                        if symbol_key not in self.data_variables:
                            msg = f"Internal compiler error: Could not find data variable {symbol_key}"
                            raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

                        symbol.variable = self.data_variables[symbol_key]

                    case ops.Param():
                        # If this symbol has a dataframe, then it must be left joinable to the primary dataframe
                        if symbol_key in self.variable_subscripts:
                            self.variable_subscripts[symbol_key]

                        # point symbol to appropriate variable
                        if symbol_key in self.assigned_parameter_variables:
                            symbol.variable = self.assigned_parameter_variables[symbol_key]
                        elif symbol_key in self.parameter_variables:
                            symbol.variable = self.parameter_variables[symbol_key]
                        else:
                            msg = f"Internal compiler error: Could not find parameter variable {symbol_key}"
                            raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

        # Build SubscriptUse variables for every time a variable is subscripted
        for top_expr in ordered_expr_trees:
            # At this point there should only be one primary variable per line
            primary_op = self._get_primary_symbol_from_statement(top_expr)

            primary_key = primary_op.get_key()

            for symbol in ops.search_tree(top_expr, ops.Data, ops.Param):
                # For each data op in the line (excluding left-hand-sides):
                # 1. Make sure that any dataframe attached to this symbol can be joined to the primary dataframe
                # 2. Link the symbol to the appropriate variable
                # 3. Build necessary SubscriptUse objects
                symbol_key = symbol.get_key()

                # resolve subscripts
                if symbol.subscript is not None:
                    primary_df = self.variable_subscripts[primary_key].base_df

                    # incorporate shifts into variable.Subscript
                    self.variable_subscripts[symbol_key].incorporate_shifts(symbol.subscript.shifts)

                    # extract the subscripts from the base df and link them into variable.SubscriptUse
                    # TODO: I'm suspicious we need some sort of error checking on this giant statement -- like
                    # I'm not sure why we expect primary_df to have the symbol subscripts here
                    variable_subscript_use = variables.SubscriptUse(
                        names=symbol.subscript.get_key(),
                        df=primary_df.loc[:, symbol.subscript.get_key()],
                        subscript=self.variable_subscripts[symbol_key],
                        unique_id=self._get_unique_identifier(),
                        shifts=symbol.subscript.shifts,
                    )

                    try:
                        variable_subscript_use.to_numpy()
                    except Exception as e:
                        msg = f"Could not uniquely join rows of dataframe of {symbol_key} into {primary_key} on {symbol_key}"
                        raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

                    # link the created variable.SubscriptUse into ops.Param
                    symbol.subscript.variable = variable_subscript_use
                    self.subscript_use_variables[variable_subscript_use.code()] = variable_subscript_use

                    # quickly check to make sure subscripts don't exist for Data
                    # for subexpr in ops.search_tree(top_expr, ops.Data):
                    #    subexpr_key = subexpr.get_key()
                    #    if subexpr.subscript:
                    #        raise CompileError(
                    #            f"Indexing on data variables ({subexpr_key}) not supported",
                    #            self.model_code_string,
                    #            subexpr.line_index,
                    #            subexpr.column_index,
                    #        )

        # If a variable is assigned on a given line, mark uses so that the code
        # generation for the scan can handle this case correctly
        for top_expr in ordered_expr_trees:
            match top_expr:
                case ops.Assignment(lhs, rhs):
                    # Assume already that the left hand side is a Param
                    assigned_key = lhs.get_key()

                    # The left hand side is written by the assignment but will be updated
                    # after the scan is complete and should not be marked
                    for symbol in ops.search_tree(rhs, ops.Param):
                        symbol_key = symbol.get_key()

                        if assigned_key == symbol_key:
                            symbol.assigned_by_scan = True

                            if symbol.subscript is None or all(shift is None for shift in symbol.subscript.shifts):
                                msg = f"Recursively assigning {symbol_key} requires a shifted subscript on the right hand side reference"
                                raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

                            # Generate the first in group indicators used to mask the scan carries
                            self.first_in_group_indicators[assigned_key] = symbol.subscript.variable.get_first_in_group_indicators()

            # self.line_functions.append(line_function)
            logging.debug(f"data: {list(self.data_variables.keys())}")
            logging.debug(f"parameters: {list(self.parameter_variables.keys())}")
            logging.debug(f"assigned parameters {list(self.assigned_parameter_variables.keys())}")

    def _generate_python_source(self):
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

        ordered_expr_trees = list(reversed(self.expr_tree_list))

        # Transform parameters
        code += "\ndef transform_parameters(data, subscripts, first_in_group_indicators, parameters):"
        for top_expr in ordered_expr_trees:
            match top_expr:
                case ops.Assignment(lhs, rhs):
                    # Assume left hand side is a parameter
                    lhs_key = top_expr.lhs.get_key()
                    code += f"\n    # {lhs_key}\n"

                    lhs_size = lhs.variable.size()
                    lhs_is_scalar = lhs_size == None
                    lhs_used_in_rhs = False
                    need_first_in_group_indicator = False
                    if not lhs_is_scalar:
                        carry_size = 0
                        # Because left hand side is parameter, we only need to search for parameters on the right hand side
                        for symbol in ops.search_tree(rhs, ops.Param):
                            symbol_key = symbol.get_key()

                            # Save an indicator if the left hand side is used on the right hand side
                            if symbol_key == lhs_key:
                                lhs_used_in_rhs = True

                                # Figure out the carry size needed
                                shifts = symbol.subscript.shifts

                                number_subscripts_shifted = sum(shift is not None for shift in shifts)
                                if number_subscripts_shifted != 1 or shifts[-1] is None:
                                    raise Exception(
                                        "Internal compiler error (shouldn't alert here): When shifting an assigned variable, can only have one shifted subscript and it must be the last"
                                    )

                                carry_size = max(carry_size, shifts[-1])

                                # There will be groupings if there is more than one shift
                                if len(shifts) > 1:
                                    need_first_in_group_indicator = True

                    if lhs_used_in_rhs:
                        # We need to scan in this case

                        if carry_size == 0:
                            raise Exception(
                                "Internal compiler error (shouldn't alert here): Carry size should be greater than zero when scan used"
                            )

                        scan_function_name = f"scan_function_{self._get_unique_identifier()}"

                        code += f"    def {scan_function_name}(carry, index):\n"
                        if need_first_in_group_indicator:
                            code += f"        carry = jax.numpy.where(first_in_group_indicators['{lhs_key}'][index], jax.numpy.zeros(carry.shape), carry)\n"
                        code += f"        next_value = {top_expr.rhs.code(scalar = True)}\n"
                        code += "        next_value_as_array = jax.numpy.array([next_value])\n"
                        code += "        return jax.numpy.concatenate([next_value_as_array, carry[1:]]), next_value\n"

                        code += f"\n    _, parameters['{lhs_key}'] = jax.lax.scan({scan_function_name}, jax.numpy.zeros({carry_size}), jax.numpy.arange({lhs_size}))\n"
                    else:
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
        Dict[str, numpy.ndarray],
        str,
    ]:
        if self.parameter_variables:
            raise Exception("Compiler.compile() may be invoked only once per instance. Create a new instance to recompile.")

        self._identify_primary_symbols()

        # build the parameter_base_df for each variable
        self._build_base_df()

        # build variable.Subscript for each parameter
        logging.debug("now printing parameter_base_df")
        for var_name, subscript_names in self.parameter_subscript_names.items():
            logging.debug(var_name)
            logging.debug(self.parameter_base_df[var_name])
            self.variable_subscripts[var_name] = variables.Subscript(self.parameter_base_df[var_name], subscript_names)
        logging.debug("done printing parameter_base_df")

        # build variable.Subscript for each data
        logging.debug("now printing data_base_df")
        for var_name, base_df in self.data_base_df.items():
            logging.debug(var_name)
            self.variable_subscripts[var_name] = variables.Subscript(base_df, [])
        logging.debug("done printing data_base_df")

        logging.debug("now printing variable_subscripts")
        for key, val in self.variable_subscripts.items():
            logging.debug(key)
            logging.debug(val.base_df)
            logging.info(f"Subscript data for parameter {key}:")
            val.log_summary(logging.INFO)
            logging.info("-----")
        logging.debug("done printing variable_subscripts")

        # run seconds pass, which binds subscripts and does codegen
        self._build_variables()

        return (
            self.data_variables,
            self.parameter_variables,
            self.assigned_parameter_variables,
            self.subscript_use_variables,
            self.first_in_group_indicators,
            self._generate_python_source(),
        )
