from ast import Lambda
from codeop import Compile
import itertools
import functools
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import List, Dict, Set, Union
import itertools

import pandas as pd

from . import ast
from . import codegen_backends
from .exceptions import CompileError, MergeError
from .variable_table import VariableTracer, VariableTable, VariableType
from rat import variable_table


class Compiler:
    data: Union[pandas.DataFrame, Dict]
    expr_tree_list: List[ast.Expr]
    model_code_string: str
    max_trace_iterations: int

    def __init__(
        self, data: Union[pandas.DataFrame, Dict], expr_tree_list: List[ast.Expr], model_code_string: str, max_trace_iterations: int
    ):
        self.data = data
        self.expr_tree_list = expr_tree_list
        self.model_code_string = model_code_string
        self.max_trace_iterations = max_trace_iterations
        self.variable_table: VariableTable = None
        self.generated_code = ""

    def _get_primary_symbol_from_statement(self, top_expr: ast.Expr):
        """
        Get the primary symbol in a statement. This assumes that the statement has
        only one primary symbol
        """
        for primary_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
            if primary_symbol.prime:
                break
        else:
            msg = f"Internal compiler error. No primary variable found"
            raise CompileError(msg, top_expr.range)
        return primary_symbol

    def _identify_primary_symbols(self):
        # figure out which symbols will have dataframes
        has_dataframe = set()
        for top_expr in self.expr_tree_list:
            for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
                if isinstance(primeable_symbol, ast.Data) or primeable_symbol.subscript is not None:
                    has_dataframe.add(primeable_symbol.get_key())

        for top_expr in self.expr_tree_list:
            # We compute the primary variable reference in a line of code with the rules:
            # 1. There can only be one primary variable reference (priming two references to the same variable is still an error)
            # 2. If a variable is marked as primary, then it is the primary variable.
            # 3. If there is no marked primary variable, then all variables with dataframes are treated as prime.
            # 4. If there are no variables with dataframes, the leftmost one is the primary one
            # 5. It is an error if no primary variable can be identified
            primary_symbol: ast.PrimeableExpr = None
            # Rule 2
            for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
                if primeable_symbol.prime:
                    if primary_symbol is None:
                        primary_symbol = primeable_symbol
                    else:
                        msg = f"Found two marked primary variables {primary_symbol.get_key()} and {primeable_symbol.get_key()}. There should only be one"
                        raise CompileError(msg, top_expr.range)

            # Rule 3
            if primary_symbol is None:
                for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
                    primeable_key = primeable_symbol.get_key()

                    if primeable_key in has_dataframe:
                        if primary_symbol is None:
                            primary_symbol = primeable_symbol
                        else:
                            primary_key = primary_symbol.get_key()
                            if primary_key != primeable_key:
                                msg = f"No marked primary variable and at least {primary_key} and {primeable_key} are candidates. A primary variable should be marked manually"
                                raise CompileError(msg, top_expr.range)
                            else:
                                msg = f"No marked primary variable but found multiple references to {primary_key}. One reference should be marked manually"
                                raise CompileError(msg, top_expr.range)

            # Rule 4
            if primary_symbol is None:
                for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
                    primary_symbol = primeable_symbol
                    break

            # Rule 5
            if primary_symbol is None:
                msg = f"No primary variable found on line (this means there are no candidate variables)"
                raise CompileError(msg, top_expr.range)

            # Mark the primary symbol if it wasn't already
            primary_symbol.prime = True

    def _get_shifts_and_padding(self, subscript: ast.Subscript):
        """
        Evaluate subscript expressions. These should all be reducible to integer shifts.

        If any subscripts are non-zero, pad_needed will be true.

        Return the integer shifts and pad_needed
        """
        # Fold shift values into integers if it's an expression instead of a constant
        integer_shifts: List[int] = []
        for shift_expr in subscript.shifts:
            match shift_expr:
                case ast.IntegerConstant():
                    integer_shifts.append(shift_expr.value)
                case _:
                    try:
                        shift_code_generator = codegen_backends.BaseCodeGenerator()
                        shift_code_generator.generate(shift_expr)
                        folded_shift = int(eval(shift_code_generator.get_expression_string()))
                        integer_shifts.append(folded_shift)
                    except Exception as e:
                        error_msg = f"Failed evaluating shift. Shift amount expressions must be an expression which can be evaluated at compile-time."
                        raise CompileError(error_msg, shift_expr.range) from e

        # If there is a non-zero shift, set pad needed to True
        pad_needed = any(shift != 0 for shift in integer_shifts)

        return tuple(integer_shifts), pad_needed

    def build_variable_table(self):
        """
        Builds the "symbol table", which holds information for all variables in the model.
        """
        self.variable_table = VariableTable(self.data)

        # Add entries to the symbol table for all data/params
        for top_expr in self.expr_tree_list:
            for symbol in top_expr:
                try:
                    match symbol:
                        case ast.Data():
                            self.variable_table.insert(variable_name=symbol.get_key(), variable_type=VariableType.DATA)
                        case ast.Param():
                            self.variable_table.insert(variable_name=symbol.get_key(), variable_type=VariableType.PARAM)
                except KeyError as e:
                    raise CompileError(str(e), symbol.range)

        # TODO: I'm not sure it's a good pattern to modify the AST in place
        # Fold all shifts to constants
        for top_expr in self.expr_tree_list:
            for symbol in top_expr:
                match symbol:
                    case ast.Data() | ast.Param():
                        if symbol.subscript is not None:
                            integer_shifts, padding_needed = self._get_shifts_and_padding(symbol.subscript)

                            new_shift_expressions = []
                            for shift_expression, integer_shift in zip(symbol.subscript.shifts, integer_shifts):
                                new_shift_expressions.append(ast.IntegerConstant(value=integer_shift, range=shift_expression.range))

                            symbol.subscript.shifts = tuple(new_shift_expressions)

        # Do a sweep to rename the primary variables that are renamed
        for top_expr in self.expr_tree_list:
            # Find the primary variable
            primary_symbol = self._get_primary_symbol_from_statement(top_expr)
            primary_symbol_key = primary_symbol.get_key()

            primary_variable = self.variable_table[primary_symbol_key]

            # Rename the primary variable
            if primary_symbol.subscript is not None:
                primary_subscript_names = tuple(column.name for column in primary_symbol.subscript.names)

                # TODO: Make property?
                try:
                    primary_variable.rename(primary_subscript_names)
                except AttributeError:
                    msg = f"Attempting to rename subscripts to {primary_subscript_names} but they have already been renamed to {primary_variable.subscripts}"
                    raise CompileError(msg, primary_symbol.range)

        # Do a sweep to infer subscript names for subscripts not renamed
        for top_expr in self.expr_tree_list:
            for symbol in top_expr:
                match symbol:
                    case ast.Param():
                        if symbol.subscript is not None:
                            symbol_key = symbol.get_key()
                            variable = self.variable_table[symbol_key]

                            subscript_names = tuple(column.name for column in symbol.subscript.names)

                            if not variable.renamed:
                                try:
                                    variable.suggest_names(subscript_names)
                                except AttributeError:
                                    msg = f"Attempting to reference subscript of {symbol_key} as {subscript_names}, but they have already been referenced as {variable.subscripts}. The subscripts must be renamed"
                                    raise CompileError(msg, symbol.range)

        # Perform a number of dataframe compatibility checks
        for top_expr in self.expr_tree_list:
            # Find the primary variable
            primary_symbol = self._get_primary_symbol_from_statement(top_expr)
            primary_symbol_key = primary_symbol.get_key()

            primary_variable = self.variable_table[primary_symbol_key]
            primary_subscript_names = primary_variable.subscripts

            for symbol in top_expr:
                match symbol:
                    # TODO: It would be nice if these code paths were closer for Data and Params
                    case ast.Data():
                        # If the target variable is data, then assume the subscripts here are
                        # referencing columns of the dataframe by name
                        symbol_key = symbol.get_key()

                        variable = self.variable_table[symbol_key]

                        if symbol.subscript is not None:
                            # Check that number/names of subscripts are compatible with primary variable
                            # and with the variable's own dataframe
                            for column in symbol.subscript.names:
                                if column.name not in primary_subscript_names:
                                    dataframe_name = self.variable_table.get_dataframe_name(primary_symbol_key)
                                    msg = f"Subscript {column.name} not found in dataframe {dataframe_name} (associated with primary variable {primary_symbol_key})"
                                    raise CompileError(msg, column.range)

                                if column.name not in variable.subscripts:
                                    dataframe_name = self.variable_table.get_dataframe_name(symbol_key)
                                    msg = f"Subscript {column.name} not found in dataframe {dataframe_name} (associated with variable {symbol_key})"
                                    raise CompileError(msg, column.range)
                    case ast.Param():
                        symbol_key = symbol.get_key()

                        variable = self.variable_table[symbol_key]

                        # If variable has no subscript, it's a scalar and there's nothing to do
                        if symbol.subscript is not None:
                            # Check that number/names of subscripts are compatible with primary variable
                            for column in symbol.subscript.names:
                                if column.name not in primary_subscript_names:
                                    msg = f"Subscript {column.name} not found in dataframe of primary variable {primary_symbol_key}"
                                    raise CompileError(msg, column.range)

                            subscript_names = [column.name for column in symbol.subscript.names]

                            # If variable is known to have subscripts then check that
                            # the number of subscripts are compatible with existing
                            # ones
                            if len(variable.subscripts) > 0:
                                if len(subscript_names) != len(variable.subscripts):
                                    msg = f"{len(subscript_names)} found, previously used with {len(variable.subscripts)}"
                                    raise CompileError(msg, symbol.range)

                            # Fold all the shift expressions to simple integers
                            shifts, pad_needed = self._get_shifts_and_padding(symbol.subscript)
                            variable.pad_needed = pad_needed

                            # Extra checks for secondary variables
                            if symbol_key != primary_symbol_key:
                                # TODO: This should probably be supported in the future
                                # right now I'm leaving it off because I'm not quite sure
                                # what the behavior should be and I don't have an example
                                # model in mind (my guess is access should give zeros)
                                if any(shift != 0 for shift in shifts):
                                    msg = f"Shifted access on a secondary parameter is not allowed"
                                    raise CompileError(msg, symbol.range)

        # Trace the program to determine parameter domains and check data domains
        tracers = {}

        for _ in range(self.max_trace_iterations):
            # Reset all the parameter tracers
            for variable_name in self.variable_table:
                variable = self.variable_table[variable_name]
                tracers[variable_name] = variable.get_tracer()

            for top_expr in self.expr_tree_list:
                # Find the primary variable
                primary_symbol = self._get_primary_symbol_from_statement(top_expr)
                primary_symbol_key = primary_symbol.get_key()

                primary_variable = self.variable_table[primary_symbol_key]

                primary_df = primary_variable.base_df
                if primary_df is not None:
                    code_generator = codegen_backends.DiscoverVariablesCodeGenerator()
                    code_generator.generate(top_expr)

                    for row in primary_df.itertuples(index=False):
                        lambda_row = { key : functools.partial(lambda x : x, value) for key, value in row._asdict().items() }
                        eval(code_generator.expression_string, globals(), {**tracers, **lambda_row})

            found_new_traces = False
            for variable_name, tracer in tracers.items():
                variable = self.variable_table[variable_name]
                if variable.variable_type == VariableType.PARAM:
                    found_new_traces |= variable.ingest_new_trace(tracer)

            if not found_new_traces:
                break
        else:
            raise CompileError(f"Unable to resolve subscripts after {self.max_trace_iterations} trace iterations")

        # Apply constraints to parameters
        for top_expr in self.expr_tree_list:
            for symbol in top_expr:
                match symbol:
                    case ast.Param():
                        symbol_key = symbol.get_key()

                        # Constraints should be evaluated at compile time
                        try:
                            lower_constraint_evaluator = codegen_backends.BaseCodeGenerator()
                            lower_constraint_evaluator.generate(symbol.lower)
                            lower_constraint_value = float(eval(lower_constraint_evaluator.get_expression_string()))

                            upper_constraint_evaluator = codegen_backends.BaseCodeGenerator()
                            upper_constraint_evaluator.generate(symbol.upper)
                            upper_constraint_value = float(eval(upper_constraint_evaluator.get_expression_string()))
                        except Exception as e:
                            error_msg = f"Failed evaluating constraints for parameter {symbol_key}, ({e})"
                            raise CompileError(error_msg, symbol.range) from e

                        variable = self.variable_table[symbol_key]
                        try:
                            variable.set_constraints(lower_constraint_value, upper_constraint_value)
                        except AttributeError:
                            msg = f"Attempting to set constraints of {symbol_key} to ({lower_constraint_value}, {upper_constraint_value}) but they are already set to ({variable.constraint_lower}, {variable.constraint_upper})"
                            raise CompileError(msg, symbol.range)

        # Apply constraints to parameters
        for top_expr in self.expr_tree_list:
            for symbol in top_expr:
                # If it's an assignment, we save the type of the LHS variable to be an assigned parameter
                match top_expr:
                    case ast.Assignment(lhs):
                        match lhs:
                            case ast.Param():
                                lhs_variable = self.variable_table[lhs.get_key()]
                                lhs_variable.variable_type = VariableType.ASSIGNED_PARAM
                            case _:
                                msg = f"The left hand of an assignment must be a parameter"
                                raise CompileError(msg, symbol.range)

    def pre_codegen_checks(self):
        N_lines = len(self.expr_tree_list)

        for line_index, top_expr in enumerate(self.expr_tree_list):
            primary_symbol = self._get_primary_symbol_from_statement(top_expr)
            primary_key = primary_symbol.get_key()

            # 1. check restrictions regarding
            match top_expr:
                case ast.Assignment(lhs, rhs):
                    # Assume already that the left hand side is a Param
                    assigned_key = lhs.get_key()

                    # The left hand side is written by the assignment but will be updated
                    # after the scan is complete and should not be marked
                    for symbol in ast.search_tree(rhs, ast.Param):
                        symbol_key = symbol.get_key()

                        if assigned_key == symbol_key:
                            symbol.assigned_by_scan = True

                            if symbol.subscript is None or all(shift is None for shift in symbol.subscript.shifts):
                                msg = f"Recursively assigning {symbol_key} requires a shifted subscript on the right hand side reference"
                                raise CompileError(msg, symbol.range)

                            shifts = [shift.value for shift in symbol.subscript.shifts]

                            if sum(shift != 0 for shift in shifts) != 1:
                                msg = "Exactly one (no more, no less) subscripts can be shifted in a recursively assigned variable"
                                raise CompileError(msg, symbol.range)

                            if shifts[-1] <= 0:
                                msg = "Only the right-most subscript of a recursively assigned variable can be shifted"
                                raise CompileError(msg, symbol.range)

                            if any(shift < 0 for shift in shifts):
                                msg = "All nonzero shifts in a recursively assigned variable must be positive"
                                raise CompileError(msg, symbol.range)

            # 2. Check that all secondary parameter uses precede primary uses (or throw an error)
            if isinstance(primary_symbol, ast.Param):
                for j in range(line_index + 1, N_lines):
                    for secondary_symbol in ast.search_tree(self.expr_tree_list[j], ast.Param):
                        secondary_key = secondary_symbol.get_key()

                        if primary_key == secondary_key and not secondary_symbol.prime:
                            msg = f"Primary variable {primary_symbol.get_key()} used on line {line_index} but then referenced as non-prime on line {j}. The primed uses must come last"
                            raise CompileError(msg, primary_symbol.range)

            # 3. Parameters cannot be used after they are assigned
            match top_expr:
                case ast.Assignment(lhs):
                    for j in range(line_index + 1, N_lines):
                        lhs_key = lhs.get_key()
                        for symbol in ast.search_tree(self.expr_tree_list[j], ast.Param):
                            symbol_key = symbol.get_key()
                            if symbol_key == lhs_key:
                                msg = f"Parameter {lhs.get_key()} is assigned on line {line_index} but used on line {j}. A variable cannot be used after it is assigned"
                                raise CompileError(msg, lhs.range)

            # 4. Parameters cannot be used in ifelse statements
            primary_record = self.variable_table[primary_key]
            allowed_subscript_names = primary_record.subscripts
            for ifelse in ast.search_tree(top_expr, ast.IfElse):
                for param in ast.search_tree(ifelse.condition, ast.Param):
                    if param.name not in allowed_subscript_names:
                        raise CompileError("Parameters are not allowed in ifelse conditions", ifelse.condition.range)

    def codegen(self):
        self.generated_code = ""
        self.generated_code += "# A rat model\n\n"
        self.generated_code += "import rat.constraints\n"
        self.generated_code += "import rat.math\n"
        self.generated_code += "import jax\n\n"

        self.generated_code += f"unconstrained_parameter_size = {self.variable_table.unconstrained_parameter_size}\n\n"

        self.codegen_constrain_parameters()

        self.codegen_transform_parameters()

        self.codegen_evaluate_densities()

    def codegen_constrain_parameters(self):
        self.generated_code += "def constrain_parameters(unconstrained_parameter_vector, pad=True):\n"
        self.generated_code += "    unconstrained_parameters = {}\n"
        self.generated_code += "    parameters = {}\n"
        self.generated_code += "    jacobian_adjustments = 0.0\n"

        for variable_name in self.variable_table:
            record = self.variable_table[variable_name]
            if record.variable_type != VariableType.PARAM:
                continue

            unconstrained_reference = f"unconstrained_parameters['{variable_name}']"
            constrained_reference = f"parameters['{variable_name}']"

            self.generated_code += "\n"
            self.generated_code += f"    # Param: {variable_name}, lower: {record.constraint_lower}, upper: {record.constraint_upper}\n"

            # This assumes that unconstrained parameter indices for a parameter is allocated in a contiguous fashion.
            if len(record.subscripts) > 0:
                index_string = f"{record.unconstrained_vector_start_index} : {record.unconstrained_vector_end_index + 1}"
            else:
                index_string = f"{record.unconstrained_vector_start_index}"

            self.generated_code += f"    {unconstrained_reference} = unconstrained_parameter_vector[..., {index_string}]\n"

            if record.constraint_lower > float("-inf") or record.constraint_upper < float("inf"):
                if record.constraint_lower > float("-inf") and record.constraint_upper == float("inf"):
                    self.generated_code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.lower({unconstrained_reference}, {record.constraint_lower})\n"
                elif record.constraint_lower == float("inf") and record.constraint_upper < float("inf"):
                    self.generated_code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.upper({unconstrained_reference}, {record.constraint_upper})\n"
                elif record.constraint_lower > float("-inf") and record.constraint_upper < float("inf"):
                    self.generated_code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.finite({unconstrained_reference}, {record.constraint_lower}, {record.constraint_upper})\n"

                self.generated_code += "    jacobian_adjustments += jax.numpy.sum(constraints_jacobian_adjustment)\n"
            else:
                self.generated_code += f"    {constrained_reference} = {unconstrained_reference}\n"

            if record.pad_needed:
                self.generated_code += "    if pad:\n"
                self.generated_code += f"        {constrained_reference} = jax.numpy.pad({constrained_reference}, (0, 1))\n"

        self.generated_code += "\n"
        self.generated_code += "    return jacobian_adjustments, parameters\n\n"

    def codegen_transform_parameters(self):
        self.generated_code += "def transform_parameters(data, subscripts, first_in_group_indicators, parameters):\n"

        for top_expr in list(reversed(self.expr_tree_list)):
            if not isinstance(top_expr, ast.Assignment):
                continue

            code_generator = codegen_backends.TransformedParametersCodeGenerator(
                self.variable_table, self._get_primary_symbol_from_statement(top_expr), indent=4
            )

            code_generator.generate(top_expr)
            self.generated_code += code_generator.get_expression_string()
            self.generated_code += "\n"

        self.generated_code += "\n"
        self.generated_code += "    return parameters\n\n"

    def codegen_evaluate_densities(self):
        self.generated_code += "def evaluate_densities(data, subscripts, parameters):\n"
        self.generated_code += "    target = 0.0\n"

        for top_expr in self.expr_tree_list:
            if not isinstance(top_expr, ast.Distr):
                continue

            code_generator = codegen_backends.EvaluateDensityCodeGenerator(
                self.variable_table, self._get_primary_symbol_from_statement(top_expr)
            )
            try:
                code_generator.generate(top_expr)
            except MergeError as e:
                msg = str(e)
                raise CompileError(msg, top_expr.range)
            self.generated_code += f"    target += jax.numpy.sum({code_generator.get_expression_string()})\n"

        self.generated_code += "\n"
        self.generated_code += "    return target\n"

    def compile(self):
        self._identify_primary_symbols()

        self.build_variable_table()

        self.variable_table.prepare_unconstrained_parameter_mapping()

        self.pre_codegen_checks()

        self.codegen()

        data_dict = {}
        base_df_dict = {}

        for variable_name in self.variable_table:
            record = self.variable_table[variable_name]
            for data_name, array in record.get_numpy_arrays():
                data_dict[data_name] = jax.numpy.array(array)
            if record.variable_type != VariableType.DATA:
                if record.base_df is not None:
                    base_df_dict[variable_name] = record.base_df
                else:
                    base_df_dict[variable_name] = pd.DataFrame()

        return (
            data_dict,
            base_df_dict,
            self.variable_table.generated_subscript_dict,
            self.variable_table.first_in_group_indicator,
            self.generated_code,
        )
