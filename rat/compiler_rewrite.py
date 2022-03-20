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


class CompileError(Exception):
    def __init__(self, message, code_string: str = "", line_num: int = -1, column_num: int = -1):
        code_string = code_string.split("\n")[line_num] if code_string else ""
        exception_message = f"An error occured while compiling the following line({line_num}:{column_num}):\n{code_string}\n{' ' * column_num + '^'}\n{message}"
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
    base_df: pd.DataFrame = field(default=None, kw_only=True)


class SymbolTable:
    def __init__(self):
        self.symbol_dict: Dict[str, TableRecord] = {}

    def upsert(self, variable_name: str, variable_type: VariableType = None, subscripts: Set[Tuple[str]] = None):
        if variable_name in self.symbol_dict:
            record = self.symbol_dict[variable_name]
            record.variable_type = variable_type
            if not subscripts:
                return
            if record.subscript_length > 0:
                for subscript_tuple in subscripts:
                    if len(subscript_tuple) != record.subscript_length:
                        raise ValueError(
                            f"Internal error - symbol table update failed. Variable {variable_name} in table has {len(record.subscripts)} subscripts, but update request has {len(subscripts)}"
                        )

            if subscripts:
                record.subscript_length = len(tuple(subscripts)[0])
                record.subscripts |= subscripts

        else:
            self.symbol_dict[variable_name] = TableRecord(variable_name, variable_type, subscripts if subscripts else set(), len(tuple(subscripts)[0]) if subscripts else 0)

    def lookup(self, variable_name: str):
        return self.symbol_dict[variable_name]

    def build_dataframes(self, data_df: pd.DataFrame):
        for variable_name, record in self.symbol_dict.items():
            if record.variable_type == VariableType.DATA:
                continue

            if record.subscript_length == 0:
                continue

            subscript_length = record.subscript_length
            base_df = pd.DataFrame()
            for subscript_tuple in record.subscripts:
                try:
                    df = data_df.loc[:, subscript_tuple]
                except KeyError as e:
                    raise CompileError(f"Internal Error - dataframe build failed. Could not index one or more columns in {subscript_tuple} from the data dataframe.") from e

                df.columns = tuple([f"subscript__{x}" for x in range(subscript_length)])
                base_df = pd.concat([base_df, df]).drop_duplicates().sort_values(list(df.columns)).reset_index(drop=True)

            record.base_df = base_df

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
                                    subscript_aliases[subscript_column.name] = {subscript_tuple[index] for subscript_tuple in primary_variable_record.subscripts}

                case _:
                    msg = f"Expected a parameter or data primary variable but got type {primary_variable.__class__.__name__}"
                    raise CompileError(msg, self.model_code_string, primary_variable.line_index, primary_variable.column_index)

            if isinstance(top_expr, ast.Assignment):
                self.symbol_table.upsert(top_expr.lhs.name, VariableType.ASSIGNED_PARAM)

            print(primary_variable_key, subscript_aliases)

            for param in ast.search_tree(top_expr, ast.Param):
                param_key = param.get_key()

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
                        raise CompileError(f"Unknown subscript name '{subscript_name}'", self.model_code_string, subscript_column.line_index, subscript_column.column_index)

                unrolled_subscripts = [tuple(x) for x in itertools.product(*subscript_list)]

                self.symbol_table.upsert(param_key, var_type, set(unrolled_subscripts))

    def generate_code(self):
        pass


    def compile(self):
        self._identify_primary_symbols()

        self.build_symbol_table()

        self.symbol_table.build_dataframes(self.data_df)