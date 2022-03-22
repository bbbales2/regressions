import collections
import itertools
import logging
import numpy
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import Iterable, List, Dict, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import pprint
import itertools
import warnings

import pandas as pd

from . import ast
from . import ir


class CompileError(Exception):
    def __init__(self, message, code_string: str = "", line_num: int = -1, column_num: int = -1):
        code_string = code_string.split("\n")[line_num] if code_string else ""
        if code_string:
            exception_message = f"An error occurred while compiling the following line({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        else:
            exception_message = f"An error occurred during compilation:\n{message}"
        super().__init__(exception_message)


class CompileWarning(UserWarning):
    def __init__(self, message, code_string: str = "", line_num: int = -1, column_num: int = -1):
        code_string = code_string.split("\n")[line_num] if code_string else ""
        if code_string:
            warning_message = f"Compilation warning({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
        else:
            warning_message = f"Compilation warning:\n{message}"
        super().__init__(warning_message)


class VariableType(Enum):
    DATA = 1
    PARAM = 2
    ASSIGNED_PARAM = 3


@dataclass()
class TableRecord:
    """
    A record within the SymbolTable

    name: Tame of the variable
    variable_type: The type of the variable(VariableType). Data, Param, or Assigned Param
    subscripts: If subscripts exist, the "true" subscript set of the variable. All subscripts aliases effectively
    point to one or more columns of the input dataframe. Those columns of the input dataframe are the true subscripts
    of the variable. If the variable has 2 subscripts, it is stored as {(column_1, column_2), ...}
    subscript_length: the length of the subscript, that is, how many subscripts are declared for the variable.
    subscript_alias: This is the "fake" subscript names, declared by the user for the variable. These values are used
    as column names of the base dataframe
    constraint_lower: value of the lower constraint
    constraint_upper: value of the upper constraint
    base_df: the calculated base dataframe. This is generated by `SymbolTable.build_base_dataframes()` and shouldn't be set
    by the programmer.
    unconstrained_vector_start_index: For parameters, the start index of the unconstrained parameter vector. Generated
    by `SymbolTable.build_base_dataframes()`.
    unconstrained_vector_end_index: For parameters, the end index of the unconstrained parameter vector. Generated by
    `SymbolTable.build_base_dataframes()`.
    """

    name: str
    variable_type: VariableType
    subscripts: Set[Tuple[str]]  # tuple of (subscript_name, subscript_name, ...)
    subscript_length: int = field(default=0)
    subscript_alias: Tuple[str] = field(default=tuple())
    constraint_lower: float = field(default=float("-inf"))
    constraint_upper: float = field(default=float("inf"))
    base_df: pd.DataFrame = field(default=None, init=False, repr=False)
    unconstrained_vector_start_index: int = field(default=None, init=False)
    unconstrained_vector_end_index: int = field(default=None, init=False)


class SymbolTable:
    def __init__(self, data_df):
        """
        symbol_dict: The internal dictionary that represents the symbol table. Key values are variable names.
        unconstrained_param_count: the length of the required unconstrained parameter vector.
        data_df: the input data dataframe
        generated_subscript_dict: Dictionary which maps subscript keys to indices.
        """
        self.symbol_dict: Dict[str, TableRecord] = {}
        self.unconstrained_param_count: int = 0  # length of the unconstrained parameter vector
        self.data_df = data_df
        self.generated_subscript_dict: Dict[str, numpy.ndarray] = {}
        self.generated_subscript_count = 0

    def upsert(
        self,
        variable_name: str,
        variable_type: VariableType = None,
        subscripts: Set[Tuple[str]] = None,
        subscript_alias: Tuple[str] = tuple(),
        constraint_lower: float = float("-inf"),
        constraint_upper: float = float("inf"),
    ):
        """
        Upsert(update or create) a record within the symbol table. The 6 arguments of this functions should be the only
        fields of the record that the programmer should provide; other fields are generated automatically by
        `build_base_dataframes()`.
        """
        if variable_name in self.symbol_dict:
            # Update fields accordingly
            record = self.symbol_dict[variable_name]
            if variable_type:
                record.variable_type = variable_type

            if subscripts:
                if record.subscript_length > 0:
                    # check that the length of subscript operations all match
                    for subscript_tuple in subscripts:
                        if len(subscript_tuple) != record.subscript_length:
                            raise ValueError(
                                f"Internal error - symbol table update failed. Variable '{variable_name}' in table has {len(record.subscripts)} subscripts, but update request has {len(subscripts)}"
                            )

                record.subscript_length = len(tuple(subscripts)[0])
                record.subscripts |= subscripts

            if subscript_alias:
                record.subscript_alias = subscript_alias

            record.constraint_lower = max(record.constraint_lower, constraint_lower)
            record.constraint_upper = min(record.constraint_upper, constraint_upper)

        else:
            # Insert if new
            self.symbol_dict[variable_name] = TableRecord(
                variable_name,
                variable_type,
                subscripts=subscripts if subscripts else set(),
                subscript_length=len(tuple(subscripts)[0]) if subscripts else 0,
                subscript_alias=subscript_alias,
                constraint_lower=constraint_lower,
                constraint_upper=constraint_upper,
            )

    def lookup(self, variable_name: str):
        """
        Dictionary indexing
        """
        return self.symbol_dict[variable_name]

    def iter_records(self):
        for name, record in self.symbol_dict.items():
            yield name, record

    def build_base_dataframes(self, data_df: pd.DataFrame):
        """
        Builds the base dataframes for parameters from the current symbol table.
        Also resets any generated subscript indices
        """
        self.generated_subscript_dict = {}
        self.generated_subscript_count = 0

        current_index = 0
        for variable_name, record in self.symbol_dict.items():
            if not record.variable_type:
                error_msg = f"Fatal internal error - variable type information for '{variable_name}' not present within the symbol table. Aborting compilation."
                raise CompileError(error_msg)

            # Data variables don't need a base df; the input df is its base dataframe.
            if record.variable_type == VariableType.DATA:
                continue

            if record.subscript_length > 0:
                base_df = pd.DataFrame()

                # Since the subscripts in the symbol table are all resolved to denote columns in the input dataframe,
                # we can directly pull out columns from the input dataframe.
                for subscript_tuple in record.subscripts:
                    try:
                        df = data_df.loc[:, subscript_tuple]
                    except KeyError as e:
                        raise CompileError(
                            f"Internal Error - dataframe build failed. Could not index one or more columns in {subscript_tuple} from the data dataframe."
                        ) from e

                    # rename base dataframe's columns to use the subscript alias
                    df.columns = tuple(record.subscript_alias)
                    base_df = pd.concat([base_df, df]).drop_duplicates().sort_values(list(df.columns)).reset_index(drop=True)



                # For parameters, allocate space on the unconstrained parameter vector and record indices
                if record.variable_type == VariableType.PARAM:
                    nrows = base_df.shape[0]
                    base_df["__index"] = pd.Series(range(current_index, current_index + nrows))
                    record.unconstrained_vector_start_index = current_index
                    record.unconstrained_vector_end_index = current_index + nrows - 1
                    current_index += nrows

                record.base_df = base_df

            else:
                if record.variable_type == VariableType.PARAM:
                    record.unconstrained_vector_start_index = current_index
                    record.unconstrained_vector_end_index = current_index
                    current_index += 1

        self.unconstrained_param_count = current_index

    def get_subscript_key(self, primary_variable_name: str, primary_subscript_names: Tuple[str], target_variable_name: str, target_subscript_names: Tuple[str], target_shift_amounts: Tuple[int]):
        """
        Find the subscript indices for (target variable, subscript, shift) given its primary variable and its declared subscript
        For example:
        param_a[s1, s2] = param_b[s1]

        Procedure:
        1. filter out base_df of param_a so that only rows containing s1 and s2 exists.
        2. filter out base_df of param_b so that only rows containing s1 exists.
        3. apply shifts to dataframe (1), creating new columns on dataframe 1
        3. merge dataframe (1) with dataframe (2) along s1, keeping indices from dataframe (2)

        """
        primary_record = self.lookup(primary_variable_name)
        target_record = self.lookup(target_variable_name)

        if primary_record.variable_type == VariableType.DATA:
            primary_base_df = self.data_df.copy()

        else:
            primary_base_df = primary_record.base_df.copy()

            for index, subscript in enumerate(primary_subscript_names):
                base_df_subscript_name = primary_record.subscript_alias[index]
                if subscript == base_df_subscript_name:
                    continue
                else:
                    subset_df = self.data_df.loc[:, [subscript]].drop_duplicates()
                    subset_df.columns = (base_df_subscript_name, )
                    primary_base_df = pd.merge(primary_base_df, subset_df, on=base_df_subscript_name, how="inner")

        primary_base_df = primary_base_df[list(target_subscript_names)]

        shift_subscripts = []
        shift_values = []
        grouping_subscripts = []
        for column, shift in zip(target_subscript_names, target_shift_amounts):
            if shift == 0:
                grouping_subscripts.append(column)
            else:
                shift_subscripts.append(column)
                shift_values.append(shift)

        if len(grouping_subscripts) > 0:
            grouped_df = primary_base_df.groupby(grouping_subscripts)
        else:
            grouped_df = primary_base_df

        for column, shift in zip(shift_subscripts, shift_values):
            print("waa")
            shifted_column = grouped_df[column].shift(shift).reset_index(drop=True)
            primary_base_df[column] = shifted_column

        target_base_df = target_record.base_df.copy().loc[:, target_record.base_df.columns != "__index"]
        target_base_df.columns = list(target_subscript_names)

        target_base_df["__in_dataframe_index"] = pd.Series(range(0, target_base_df.shape[0]))


        key_name = f"subscript__{self.generated_subscript_count}"
        self.generated_subscript_count += 1
        self.generated_subscript_dict[key_name] = pd.merge(primary_base_df[list(target_subscript_names)], target_base_df, on=target_subscript_names, how="left")["__in_dataframe_index"].to_numpy(dtype=int)
        return key_name


    def __str__(self):
        return pprint.pformat(self.symbol_dict)


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
        self.base_df: Dict[str, pd.DataFrame] = {}
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
            for param in ast.search_tree(top_expr, ast.Param):
                param_key = param.get_key()

                # If there are constraints, we evaluate their values and store them on the symbol table
                try:
                    lower_constraint_evaluator = ir.BaseVisitor(self.symbol_table)
                    param.lower.accept(lower_constraint_evaluator)
                    lower_constraint_value = float(eval(lower_constraint_evaluator.expression_string))

                    upper_constraint_evaluator = ir.BaseVisitor(self.symbol_table)
                    param.upper.accept(upper_constraint_evaluator)
                    upper_constraint_value = float(eval(upper_constraint_evaluator.expression_string))
                except Exception as e:
                    error_msg = f"Failed evaluating constraints for parameter {param_key}"
                    raise CompileError(error_msg, self.model_code_string, primary_variable.line_index, primary_variable.column_index) from e
                else:
                    self.symbol_table.upsert(
                        variable_name=param_key, constraint_lower=lower_constraint_value, constraint_upper=upper_constraint_value
                    )
                ###########

                # The following operations are done for subscripts only
                if not param.subscript:
                    continue

                # if param_key == primary_variable_key:
                #     continue

                # fold shift values into IntegerConstant if it's an expression instead of a constant
                folded_shift_exprs: List[ast.IntegerConstant] = []
                for shift_expr in param.subscript.shifts:
                    if isinstance(shift_expr, ast.IntegerConstant):
                        folded_shift_exprs.append(shift_expr)
                    else:
                        try:
                            shift_code_visitor = ir.BaseVisitor(self.symbol_table)
                            shift_expr.accept(shift_code_visitor)
                            folded_integerconstant = ast.IntegerConstant(value=int(eval(shift_code_visitor.expression_string)))
                            folded_shift_exprs.append(folded_integerconstant)
                        except Exception as e:
                            error_msg = f"Failed evaluating shift amounts for parameter {param_key}. Shift amount expressions must be an expression which can be evaluated at compile-time."
                            raise CompileError(error_msg, self.model_code_string, shift_expr.line_index, shift_expr.column_index) from e
                param.subscript.shifts = folded_shift_exprs
                ###########

                # Determine type of the variable. If its name exists in the input dataframe, it's a data variable.
                # Param is the default value. Assigned params are upserted when the compiler identifies it being used on
                # the LHS of an assignment.
                try:
                    var_type = self.symbol_table.lookup(param_key).variable_type
                except KeyError:
                    var_type = VariableType.DATA if param_key in self.data_df_columns else VariableType.PARAM
                ###########

                # We "resolve" subscripts, so every single parameters' subscripts can be described with only the input
                # dataframe's columns. The symbol table holds the "true" subscript columns for each parameter.
                n_subscripts = len(param.subscript.names)
                subscript_list = [[] for _ in range(n_subscripts)]

                variable_subscript_alias: List[str] = []  # This list holds the actual column names of the base DF
                for index in range(n_subscripts):
                    subscript_column = param.subscript.names[index]
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
                # ex) sub_1 = {sub_2, sub_3, sub_4}
                # then the true subscripts of param[sub_1, x] is [(sub_2, x), (sub_3, x), (sub_4, x)]
                unrolled_subscripts = [tuple(x) for x in itertools.product(*subscript_list)]

                self.symbol_table.upsert(
                    param_key, var_type, subscripts=set(unrolled_subscripts), subscript_alias=tuple(variable_subscript_alias)
                )


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
        self.generated_code += "def constrain_parameters(unconstrained_parameter_vector):\n"
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
                if record.unconstrained_vector_start_index != record.unconstrained_vector_end_index
                else f"{record.unconstrained_vector_start_index}"
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

        self.generated_code += "\n"
        self.generated_code += "    return jacobian_adjustments, parameters\n\n"

    def codegen_transform_parameters(self):
        self.generated_code += "def transform_parameters(data, subscripts, first_in_group_indicators, parameters)\n"

        for top_expr in self.expr_tree_list:
            if not isinstance(top_expr, ast.Assignment):
                continue

            codegen_visitor = ir.TransformedParametersVisitor(self.symbol_table, self._get_primary_symbol_from_statement(top_expr))
            top_expr.accept(codegen_visitor)
            self.generated_code += "    "
            self.generated_code += codegen_visitor.expression_string

        self.generated_code += "\n\n"
        self.generated_code += "    return parameters\n\n"

    def codegen_evaluate_densities(self):
        self.generated_code += "def evaluate_densities(data, subscripts, parameters):\n"
        self.generated_code += "    target = 0.0\n"

        for top_expr in self.expr_tree_list:
            if not isinstance(top_expr, ast.Distr):
                continue

            codegen_visitor = ir.EvaluateDensityVisitor(self.symbol_table, self._get_primary_symbol_from_statement(top_expr))
            top_expr.accept(codegen_visitor)
            self.generated_code += f"    target += jax.numpy.sum({codegen_visitor.expression_string})\n"

        self.generated_code += "\n"
        self.generated_code += "    return target\n"

    def compile(self):
        self._identify_primary_symbols()

        self.build_symbol_table()

        self.symbol_table.build_base_dataframes(self.data_df)

        print(self.symbol_table)

        self.codegen()
