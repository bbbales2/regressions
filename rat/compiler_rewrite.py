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
import pprint
import itertools

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


class VariableType(Enum):
    DATA = 1
    PARAM = 2
    ASSIGNED_PARAM = 3


@dataclass()
class TableRecord:
    name: str
    variable_type: VariableType
    subscripts: Set[Tuple[str]]  # tuple of (subscript_name, subscript_name, ...)
    subscript_length: int = field(default=0)
    constraint_lower: float = field(default=float("-inf"))
    constraint_upper: float = field(default=float("inf"))
    base_df: pd.DataFrame = field(default=None, init=False, repr=False)
    unconstrained_vector_start_index: int = field(default=-1, init=False)
    unconstrained_vector_end_index: int = field(default=-1, init=False)


class SymbolTable:
    def __init__(self):
        self.symbol_dict: Dict[str, TableRecord] = {}
        self.unconstrained_param_count: int = 0

    def upsert(
        self,
        variable_name: str,
        variable_type: VariableType = None,
        subscripts: Set[Tuple[str]] = None,
        constraint_lower: float = float("-inf"),
        constraint_upper: float = float("inf"),
    ):
        if variable_name in self.symbol_dict:
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

            record.constraint_lower = max(record.constraint_lower, constraint_lower)
            record.constraint_upper = min(record.constraint_upper, constraint_upper)

        else:
            self.symbol_dict[variable_name] = TableRecord(
                variable_name,
                variable_type,
                subscripts if subscripts else set(),
                len(tuple(subscripts)[0]) if subscripts else 0,
                constraint_lower=constraint_lower,
                constraint_upper=constraint_upper,
            )

    def lookup(self, variable_name: str):
        return self.symbol_dict[variable_name]

    def build_dataframes(self, data_df: pd.DataFrame):
        current_index = 0
        for variable_name, record in self.symbol_dict.items():
            if not record.variable_type:
                error_msg = f"Fatal internal error - variable type information for '{variable_name}' not present within the symbol table. Aborting compilation."
                raise CompileError(error_msg)

            if record.variable_type == VariableType.DATA:
                continue

            if record.subscript_length > 0:
                subscript_length = record.subscript_length
                base_df = pd.DataFrame()
                for subscript_tuple in record.subscripts:
                    try:
                        df = data_df.loc[:, subscript_tuple]
                    except KeyError as e:
                        raise CompileError(
                            f"Internal Error - dataframe build failed. Could not index one or more columns in {subscript_tuple} from the data dataframe."
                        ) from e

                    df.columns = tuple([f"subscript__{x}" for x in range(subscript_length)])
                    base_df = pd.concat([base_df, df]).drop_duplicates().sort_values(list(df.columns)).reset_index(drop=True)

                if record.variable_type == VariableType.PARAM:
                    nrows = base_df.shape[0]
                    base_df["index__"] = pd.Series(range(current_index, current_index + nrows))
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

    def get_parameter_indices(self, primary_variable_name, target_variable_name):

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
        self.symbol_table = SymbolTable()

        for top_expr in self.expr_tree_list:
            primary_variable = self._get_primary_symbol_from_statement(top_expr)
            primary_variable_key = primary_variable.get_key()

            if isinstance(primary_variable, ast.Data):
                self.symbol_table.upsert(variable_name=primary_variable.name, variable_type=VariableType.DATA)
            elif isinstance(primary_variable, ast.Param):
                self.symbol_table.upsert(variable_name=primary_variable.name, variable_type=VariableType.PARAM)

            subscript_aliases: Dict[str, Set] = {}
            # this dict stores any subscript aliases of the primary dataframe.

            match primary_variable:
                case ast.Data():
                    pass
                case ast.Param():
                    try:
                        primary_variable_record = self.symbol_table.lookup(primary_variable_key)
                    except KeyError:
                        msg = f"Primary variable {primary_variable_key} does not exist on the symbol table. Did you define the variable before using it as a primary variable?"
                        raise CompileError(msg, self.model_code_string, primary_variable.line_index, primary_variable.column_index)
                    else:
                        if primary_variable.subscript:
                            for index, subscript_column in enumerate(primary_variable.subscript.names):
                                if subscript_column.name not in self.data_df_columns:
                                    subscript_aliases[subscript_column.name] = {
                                        subscript_tuple[index] for subscript_tuple in primary_variable_record.subscripts
                                    }

                case _:
                    msg = f"Expected a parameter or data primary variable but got type {primary_variable.__class__.__name__}"
                    raise CompileError(msg, self.model_code_string, primary_variable.line_index, primary_variable.column_index)

            if isinstance(top_expr, ast.Assignment):
                self.symbol_table.upsert(variable_name=top_expr.lhs.name, variable_type=VariableType.ASSIGNED_PARAM)

            for param in ast.search_tree(top_expr, ast.Param):
                param_key = param.get_key()

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

                if not param.subscript:
                    continue

                if param_key == primary_variable_key:
                    continue

                try:
                    var_type = self.symbol_table.lookup(param_key).variable_type
                except KeyError:
                    var_type = VariableType.DATA if param_key in self.data_df_columns else VariableType.PARAM

                n_subscripts = len(param.subscript.names)
                subscript_list = [[] for _ in range(n_subscripts)]
                for index in range(n_subscripts):
                    subscript_column = param.subscript.names[index]
                    subscript_name = subscript_column.name

                    if subscript_name in self.data_df_columns:
                        subscript_list[index].append(subscript_name)
                    elif subscript_name in subscript_aliases:
                        subscript_list[index].extend(subscript_aliases[subscript_name])
                    else:
                        raise CompileError(
                            f"Unknown subscript name '{subscript_name}'",
                            self.model_code_string,
                            subscript_column.line_index,
                            subscript_column.column_index,
                        )

                unrolled_subscripts = [tuple(x) for x in itertools.product(*subscript_list)]

                self.symbol_table.upsert(param_key, var_type, set(unrolled_subscripts))

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
                f"{record.unconstrained_vector_start_index} : {record.unconstrained_vector_end_index}"
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
        pass

    def codegen_evaluate_densities(self):
        self.generated_code += "def evaluate_densities(data, subscripts, parameters):\n"
        self.generated_code += "target = 0.0"

        for top_expr in self.expr_tree_list:
            if not isinstance(top_expr, ast.Distr):
                continue

            codegen_visitor = ir.BaseVisitor(self.symbol_table)
            top_expr.accept(codegen_visitor)
            self.generated_code += f"    target += jax.numpy.sum({codegen_visitor.expression_string})\n"

        self.generated_code += "\n"
        self.generated_code += "    return target\n"

    def compile(self):
        self._identify_primary_symbols()

        self.build_symbol_table()

        self.symbol_table.build_dataframes(self.data_df)

        self.codegen()
