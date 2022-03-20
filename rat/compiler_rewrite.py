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
from dataclasses import dataclass
import pprint
import warnings

from . import ast


class CompileError(Exception):
    def __init__(self, message, code_string: str, line_num: int = -1, column_num: int = -1):
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
    subscripts: List[Set[Tuple[str, int]]]  # tuple of (subscript_name, shift_amount)


class SymbolTable:
    def __init__(self):
        self.symbol_dict: Dict[str, TableRecord] = {}

    def upsert(self, variable_name: str, variable_type: VariableType = None, subscripts: List[Set[Tuple[str, int]]] = None):
        if variable_name in self.symbol_dict:
            record = self.symbol_dict[variable_name]
            record.variable_type = variable_type
            if not subscripts:
                return
            if len(record.subscripts) != len(subscripts) and len(record.subscripts) > 0:
                raise ValueError(f"Internal error - symbol table update failed. Variable {variable_name} in table has {len(record.subscripts)} subscripts, but update request has {len(subscripts)}")

            for index, subscript_set in enumerate(subscripts):
                try:
                    record.subscripts[index] |= subscript_set
                except IndexError:
                    record.subscripts.append(subscript_set)

        else:
            self.symbol_dict[variable_name] = TableRecord(variable_name, variable_type, subscripts)

    def lookup(self, variable_name: str):
        return self.symbol_dict[variable_name]

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
                                    subscript_aliases[subscript_column.name] = primary_variable_record.subscripts[index]

                case _:
                    msg = f"Expected a parameter or data primary variable but got type {primary_variable.__class__.__name__}"
                    raise CompileError(msg, self.model_code_string, primary_variable.line_index,
                                       primary_variable.column_index)

            if isinstance(top_expr, ast.Assignment):
                self.symbol_table.upsert(top_expr.lhs.name, VariableType.ASSIGNED_PARAM)

            for param in ast.search_tree(top_expr, ast.Param):
                param_key = param.get_key()

                if param_key == primary_variable_key:
                    continue

                if not param.subscript:
                    continue

                try:
                    var_type = self.symbol_table.lookup(param_key).variable_type
                except KeyError:
                    var_type = VariableType.DATA if param_key in self.data_df_columns else VariableType.PARAM
                n_subscripts = len(param.subscript.names)
                subscript_list = []
                for index in range(n_subscripts):
                    subscript_name = param.subscript.names[index].name
                    subscript_shift = param.subscript.shifts[index].value

                    if subscript_name in self.data_df_columns:
                        subscript_list.append({(subscript_name, subscript_shift)})
                    elif subscript_name in subscript_aliases:
                        subscript_list.append({s for s in subscript_aliases[subscript_name]})
                    else:
                        print("ERROR ERROR", subscript_name)


                self.symbol_table.upsert(param_key, var_type, subscript_list)

        print(self.symbol_table)

    def compile(self):
        self._identify_primary_symbols()

        self.build_symbol_table()