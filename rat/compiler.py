import itertools
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import List, Dict, Set
import itertools

import pandas as pd

from . import ast
from . import codegen_backends
from .exceptions import CompileError
from .symbol_table import SymbolTable, VariableType


class Compiler:
    data_df: pandas.DataFrame
    expr_tree_list: List[ast.Expr]
    model_code_string: str

    def __init__(self, data_df: pandas.DataFrame, expr_tree_list: List[ast.Expr], model_code_string: str = ""):
        self.data_df = data_df
        self.data_df_columns = list(data_df.columns)
        self.expr_tree_list = expr_tree_list
        self.model_code_string = model_code_string
        self.symbol_table: SymbolTable = None
        self.generated_code = ""

    def _get_primary_symbol_from_statement(self, top_expr):
        """
        Get the primary symbol in a statement. This assumes that the statement has
        only one primary symbol
        """
        for primary_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
            if primary_symbol.prime:
                break
        else:
            msg = f"Internal compiler error. No primary variable found"
            raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)
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
                        raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)

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
                                raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)
                            else:
                                msg = f"No marked primary variable but found multiple references to {primary_key}. One reference should be marked manually"
                                raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)

            # Rule 4
            if primary_symbol is None:
                for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
                    primary_symbol = primeable_symbol
                    break

            # Rule 5
            if primary_symbol is None:
                msg = f"No primary variable found on line (this means there are no candidate variables)"
                raise CompileError(msg, self.model_code_string, top_expr.line_index, top_expr.column_index)

            # Mark the primary symbol if it wasn't already
            primary_symbol.prime = True

    def build_symbol_table(self):
        """
        Builds the "symbol table", which holds information for all variables in the model.
        """
        self.symbol_table = SymbolTable(self.data_df)

        for top_expr in self.expr_tree_list:
            if isinstance(top_expr, ast.Distr):
                lhs_variable_key = top_expr.variate.get_key()
            elif isinstance(top_expr, ast.Assignment):
                lhs_variable_key = top_expr.lhs.get_key()

            primary_variable = self._get_primary_symbol_from_statement(top_expr)
            primary_variable_key = primary_variable.get_key()

            if isinstance(primary_variable, ast.Data):
                self.symbol_table.upsert(variable_name=primary_variable.name, variable_type=VariableType.DATA)
            elif isinstance(primary_variable, ast.Param):
                self.symbol_table.upsert(variable_name=primary_variable.name, variable_type=VariableType.PARAM)

            subscript_aliases: Dict[str, Set] = {}
            # this dict stores any subscript aliases of the primary dataframe.

            # find the primary variable and its subscript aliases
            match primary_variable:
                case ast.Data():
                    # these are the allowed subscript names in scope
                    allowed_subscript_names = tuple(self.data_df.columns)
                case ast.Param():
                    try:
                        primary_variable_record = self.symbol_table.lookup(primary_variable_key)
                    except KeyError:
                        msg = f"Primary variable {primary_variable_key} does not exist on the symbol table. Did you define the variable before using it as a primary variable?"
                        raise CompileError(msg, self.model_code_string, primary_variable.line_index, primary_variable.column_index)
                    else:
                        if primary_variable_record.subscript_length > 0:
                            # these are the allowed subscript names in scope

                            if primary_variable.subscript:
                                for index, subscript_column in enumerate(primary_variable.subscript.names):
                                    if subscript_column.name not in self.data_df_columns:
                                        subscript_aliases[subscript_column.name] = {
                                            subscript_tuple[index] for subscript_tuple in primary_variable_record.subscripts
                                        }

                                self.symbol_table.upsert(
                                    primary_variable_key, subscript_alias=tuple([x.name for x in primary_variable.subscript.names])
                                )

                            else:
                                for index, subscript_column in enumerate(primary_variable_record.subscript_alias):
                                    if subscript_column not in self.data_df_columns:
                                        # If there's a subscript name that's not present in the input dataframe, this means
                                        # it's a subscript "alias", which points to the "true" subscripts that's already on
                                        # the symbol table. We look up its "true" subscripts and keep them in the scope of
                                        # this line
                                        subscript_aliases[subscript_column] = {
                                            subscript_tuple[index] for subscript_tuple in primary_variable_record.subscripts
                                        }
                            allowed_subscript_names = primary_variable_record.subscript_alias
                        else:
                            # these are the allowed subscript names in scope
                            allowed_subscript_names = tuple()

                case _:
                    msg = f"Expected a parameter or data primary variable but got type {primary_variable.__class__.__name__}"
                    raise CompileError(msg, self.model_code_string, primary_variable.line_index, primary_variable.column_index)

            # If it's an assignment, we save the type of the LHS variable to be an assigned parameter
            if isinstance(top_expr, ast.Assignment):
                self.symbol_table.upsert(variable_name=top_expr.lhs.name, variable_type=VariableType.ASSIGNED_PARAM)

            # iterate through the remaining non-primary variables, resolving subscripts according to primary's subscripts
            for variable in ast.search_tree(top_expr, ast.Param, ast.Data):
                variable_key = variable.get_key()

                if isinstance(variable, ast.Data):
                    self.symbol_table.upsert(variable_name=variable_key, variable_type=VariableType.DATA)
                    continue

                # If there are constraints, we evaluate their values and store them on the symbol table
                try:
                    lower_constraint_evaluator = codegen_backends.BaseCodeGenerator(self.symbol_table)
                    lower_constraint_evaluator.generate(variable.lower)
                    lower_constraint_value = float(eval(lower_constraint_evaluator.get_expression_string()))

                    upper_constraint_evaluator = codegen_backends.BaseCodeGenerator(self.symbol_table)
                    upper_constraint_evaluator.generate(variable.upper)
                    upper_constraint_value = float(eval(upper_constraint_evaluator.get_expression_string()))
                except Exception as e:
                    error_msg = f"Failed evaluating constraints for parameter {variable_key}, ({e})"
                    raise CompileError(error_msg, self.model_code_string, primary_variable.line_index, primary_variable.column_index) from e
                else:
                    self.symbol_table.upsert(
                        variable_name=variable_key, constraint_lower=lower_constraint_value, constraint_upper=upper_constraint_value
                    )
                ###########

                # The following operations are done for subscripts only
                if not variable.subscript:
                    continue

                # fold shift values into IntegerConstant if it's an expression instead of a constant
                folded_shift_exprs: List[ast.IntegerConstant] = []
                for shift_expr in variable.subscript.shifts:
                    if isinstance(shift_expr, ast.IntegerConstant):
                        folded_shift_exprs.append(shift_expr)
                    else:
                        try:
                            shift_code_generator = codegen_backends.BaseCodeGenerator(self.symbol_table)
                            shift_code_generator.generate(shift_expr)
                            folded_integerconstant = ast.IntegerConstant(value=int(eval(shift_code_generator.get_expression_string())))
                            folded_shift_exprs.append(folded_integerconstant)

                        except Exception as e:
                            error_msg = f"Failed evaluating shift amounts for parameter {variable_key}. Shift amount expressions must be an expression which can be evaluated at compile-time."
                            raise CompileError(error_msg, self.model_code_string, shift_expr.line_index, shift_expr.column_index) from e

                # If there is shift, and isn't a recursively set parameter, set pad needed to True

                # if lhs_variable_key == variable_key and isinstance(top_expr, ast.Assignment):
                #     self.symbol_table.upsert(variable_name=variable_key, pad_needed=False)
                # else:
                #     for integer_constant in folded_shift_exprs:
                #         if integer_constant.value != 0:
                #             if not self.symbol_table.lookup(variable_key).pad_needed:
                #                 self.symbol_table.upsert(variable_name=variable_key, pad_needed=True)
                #             break
                for integer_constant in folded_shift_exprs:
                    if integer_constant.value != 0:
                        if not self.symbol_table.lookup(variable_key).pad_needed:
                            self.symbol_table.upsert(variable_name=variable_key, pad_needed=True)
                        break
                variable.subscript.shifts = folded_shift_exprs
                ###########

                # Determine type of the variable. If its name exists in the input dataframe, it's a data variable.
                # Param is the default value. Assigned params are upserted when the compiler identifies it being used on
                # the LHS of an assignment.
                try:
                    var_type = self.symbol_table.lookup(variable_key).variable_type
                except KeyError:
                    var_type = VariableType.DATA if variable_key in self.data_df_columns else VariableType.PARAM
                ###########

                # We "resolve" subscripts, so every single parameters' subscripts can be described with only the input
                # dataframe's columns. The symbol table holds the "true" subscript columns for each parameter.
                n_subscripts = len(variable.subscript.names)
                subscript_list = [[] for _ in range(n_subscripts)]

                variable_subscript_alias: List[str] = []  # This list holds the actual column names of the base DF
                for index in range(n_subscripts):
                    subscript_column = variable.subscript.names[index]
                    subscript_name = subscript_column.name

                    if subscript_name in subscript_aliases:
                        subscript_list[index].extend(subscript_aliases[subscript_name])
                    elif subscript_name in allowed_subscript_names:
                        subscript_list[index].append(subscript_name)
                    else:
                        raise CompileError(
                            f"Subscript name '{subscript_name}' not in scope.",
                            self.model_code_string,
                            subscript_column.line_index,
                            subscript_column.column_index,
                        )
                    variable_subscript_alias.append(subscript_name)

                # store the "true" subscripts
                # ex) let sub_1 := {sub_2, sub_3, sub_4}
                # then the true subscripts of param[sub_1, x] is [(sub_2, x), (sub_3, x), (sub_4, x)],
                # with subscript length 2
                unrolled_subscripts = [tuple(x) for x in itertools.product(*subscript_list)]

                self.symbol_table.upsert(
                    variable_key, var_type, subscripts=set(unrolled_subscripts), subscript_alias=tuple(variable_subscript_alias)
                )

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
                                raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

                            shifts = [shift.value for shift in symbol.subscript.shifts]

                            if sum(shift != 0 for shift in shifts) != 1:
                                msg = "Exactly one (no more, no less) subscripts can be shifted in a recursively assigned variable"
                                raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

                            if shifts[-1] <= 0:
                                msg = "Only the right-most subscript of a recursively assigned variable can be shifted"
                                raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

                            if any(shift < 0 for shift in shifts):
                                msg = "All nonzero shifts in a recursively assigned variable must be positive"
                                raise CompileError(msg, self.model_code_string, symbol.line_index, symbol.column_index)

            # 2. Check that all secondary parameter uses precede primary uses (or throw an error)
            if isinstance(primary_symbol, ast.Param):
                for j in range(line_index + 1, N_lines):
                    for secondary_symbol in ast.search_tree(self.expr_tree_list[j], ast.Param):
                        secondary_key = secondary_symbol.get_key()

                        if primary_key == secondary_key and not secondary_symbol.prime:
                            msg = f"Primary variable {primary_symbol.get_key()} used on line {line_index} but then referenced as non-prime on line {j}. The primed uses must come last"
                            raise CompileError(msg, self.model_code_string, primary_symbol.line_index, primary_symbol.column_index)

            # 3. Parameters cannot be used after they are assigned
            match top_expr:
                case ast.Assignment(lhs):
                    for j in range(line_index + 1, N_lines):
                        lhs_key = lhs.get_key()
                        for symbol in ast.search_tree(self.expr_tree_list[j], ast.Param):
                            symbol_key = symbol.get_key()
                            if symbol_key == lhs_key:
                                msg = f"Parameter {lhs.get_key()} is assigned on line {line_index} but used on line {j}. A variable cannot be used after it is assigned"
                                raise CompileError(msg, self.model_code_string, lhs.line_index, lhs.column_index)

    def codegen(self):
        self.generated_code = ""
        self.generated_code += "# A rat model\n\n"
        self.generated_code += "import rat.constraints\n"
        self.generated_code += "import rat.math\n"
        self.generated_code += "import jax\n\n"

        self.generated_code += f"unconstrained_parameter_size = {self.symbol_table.unconstrained_param_count}\n\n"

        self.codegen_constrain_parameters()

        self.codegen_transform_parameters()

        self.codegen_evaluate_densities()

    def codegen_constrain_parameters(self):
        self.generated_code += "def constrain_parameters(unconstrained_parameter_vector, pad=True):\n"
        self.generated_code += "    unconstrained_parameters = {}\n"
        self.generated_code += "    parameters = {}\n"
        self.generated_code += "    jacobian_adjustments = 0.0\n"

        for variable_name, record in self.symbol_table.symbol_dict.items():
            if record.variable_type != VariableType.PARAM:
                continue

            unconstrained_reference = f"unconstrained_parameters['{variable_name}']"
            constrained_reference = f"parameters['{variable_name}']"

            self.generated_code += "\n"
            self.generated_code += f"    # Param: {variable_name}, lower: {record.constraint_lower}, upper: {record.constraint_upper}\n"

            # This assumes that unconstrained parameter indices for a parameter is allocated in a contiguous fashion.
            index_string = (
                f"{record.unconstrained_vector_start_index} : {record.unconstrained_vector_end_index + 1}"
                if record.subscript_length > 0
                else f"[{record.unconstrained_vector_start_index}]"
            )
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
                self.symbol_table, self._get_primary_symbol_from_statement(top_expr), indent=4
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
                self.symbol_table, self._get_primary_symbol_from_statement(top_expr)
            )
            code_generator.generate(top_expr)
            self.generated_code += f"    target += jax.numpy.sum({code_generator.get_expression_string()})\n"

        self.generated_code += "\n"
        self.generated_code += "    return target\n"

    def compile(self):
        self._identify_primary_symbols()

        self.build_symbol_table()

        self.symbol_table.build_base_dataframes(self.data_df)

        self.pre_codegen_checks()

        self.codegen()

        data_dict = {}
        base_df_dict = {}

        for variable_name, record in self.symbol_table.iter_records():
            if record.variable_type == VariableType.DATA:
                data_dict[variable_name] = jax.numpy.array(self.data_df[variable_name].to_numpy())
            else:
                if record.base_df is not None:
                    base_df_dict[variable_name] = record.base_df
                else:
                    base_df_dict[variable_name] = pd.DataFrame()

        return (
            data_dict,
            base_df_dict,
            self.symbol_table.generated_subscript_dict,
            self.symbol_table.first_in_group_indicator,
            self.generated_code,
        )
