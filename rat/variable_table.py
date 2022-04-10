from dataclasses import dataclass, field
from enum import Enum
import numpy
import pandas
import pprint
from typing import Dict, Set, Tuple, Iterable

from .exceptions import CompileError


class VariableType(Enum):
    DATA = 1
    PARAM = 2
    ASSIGNED_PARAM = 3


@dataclass()
class VariableRecord:
    """
    A record within the VariableTable

    name: Name of the variable
    variable_type: Variable type
    base_df: The current base dataframe

    subscript_rename: If the variable is renamed, save the rename here (otherwise None)

    constraint_lower: Value of the lower constraint
    constraint_upper: Value of the upper constraint
    pad_needed: True if a padding is needed for the variable

    unconstrained_vector_start_index: For parameters, the start index of the unconstrained parameter vector
    unconstrained_vector_end_index: For parameters, the end index of the unconstrained parameter vector
    """

    name: str
    variable_type: VariableType
    base_df: pandas.DataFrame

    subscripts_rename: Tuple[str] = field(default=None)

    constraint_lower: float = field(default=float("-inf"))
    constraint_upper: float = field(default=float("inf"))
    pad_needed: bool = field(default=False)

    unconstrained_vector_start_index: int = field(default=None, init=False)
    unconstrained_vector_end_index: int = field(default=None, init=False)

    @property
    def subscripts(self):
        if self.subscripts_rename is None:
            if self.base_df is None:
                return ()
            else:
                return tuple(self.base_df.columns)
        else:
            return self.subscripts_rename

    def set_subscript_names(self, subscript_rename: Tuple[str]):
        if self.subscripts_rename is None:
            self.subscripts_rename = subscript_rename
            self.base_df.columns = self.subscripts
        else:
            if subscript_rename != self.subscripts_rename:
                raise AttributeError("Internal compiler error: If there are multiple renames, the rename values must match")

    def add_rows_from_dataframe(self, new_rows_df: pandas.DataFrame):
        """
        Add rows to the base_df
        """
        if self.base_df is None:
            combined_df = new_rows_df
        else:
            combined_df = pandas.DataFrame(numpy.concatenate([self.base_df.values, new_rows_df.values]), columns=self.base_df.columns)

        self.base_df = combined_df.drop_duplicates().sort_values(list(combined_df.columns)).reset_index(drop=True)

    def set_constraints(self, constraint_lower: float, constraint_upper: float):
        if self.constraint_lower != float("-inf") or self.constraint_upper != float("inf"):
            if self.constraint_lower != constraint_lower or self.constraint_upper != constraint_upper:
                raise AttributeError("Internal compiler error: Once changed from defaults, constraints must match")
        else:
            self.constraint_lower = constraint_lower
            self.constraint_upper = constraint_upper


class VariableTable:
    symbol_dict : Dict[str, VariableRecord]
    unconstrained_parameter_size : int
    data_df : pandas.DataFrame

    generated_subscript_dict : Dict[str, numpy.ndarray]
    first_in_group_indicator : Dict[str, numpy.ndarray]

    _unique_number : int

    def __init__(self, data_df : pandas.DataFrame):
        self.symbol_dict = {}
        self.unconstrained_parameter_size = None
        self.data_df = data_df.copy()

        # Convert all integer columns to a type that supports NA
        for column in self.data_df.columns:
            if pandas.api.types.is_integer_dtype(self.data_df[column].dtype):
                self.data_df[column] = self.data_df[column].astype(pandas.Int64Dtype())

        self.generated_subscript_dict = {}
        self.first_in_group_indicator = {}

        self._unique_number = 0

    def get_unique_number(self):
        self._unique_number += 1
        return self._unique_number

    def insert(
        self,
        variable_name: str,
        variable_type: VariableType = None,
    ):
        """
        Insert a variable record. If it is data, attach the input dataframe to it
        """
        if variable_name not in self.symbol_dict:
            if variable_type == VariableType.DATA:
                base_df = self.data_df
            else:
                base_df = None

            # Insert if new
            self.symbol_dict[variable_name] = VariableRecord(
                variable_name,
                variable_type,
                base_df,
            )

    def __contains__(self, name: str) -> bool:
        return name in self.symbol_dict

    def __getitem__(self, variable_name: str) -> VariableRecord:
        return self.symbol_dict[variable_name]

    def __iter__(self):
        for name in self.symbol_dict:
            yield name

    def prepare_unconstrained_parameter_mapping(self):
        """
        Compute the size of vector that can hold everything and figure out
        how each record maps into this vector
        """
        current_index = 0

        for variable_name, record in self.symbol_dict.items():
            if not record.variable_type:
                error_msg = f"Fatal internal error - variable type information for '{variable_name}' not present within the symbol table. Aborting compilation."
                raise CompileError(error_msg)

            # Allocate space on unconstrained parameter vector for parameters
            if record.variable_type == VariableType.PARAM:
                if len(record.subscripts) > 0:
                    nrows = record.base_df.shape[0]
                else:
                    nrows = 1

                record.unconstrained_vector_start_index = current_index
                record.unconstrained_vector_end_index = current_index + nrows - 1
                current_index += nrows
        
        self.unconstrained_parameter_size = current_index


    def get_subscript_key(
        self,
        primary_variable_name: str,
        target_variable_name: str,
        target_variable_subscripts: Tuple[str],
        shifts: Tuple[int],
    ) -> str:
        primary_variable = self[primary_variable_name]
        target_variable = self[target_variable_name]

        shifts_by_subscript_name = {
            name: shift for name, shift in zip(target_variable_subscripts, shifts)
        }

        target_variable.base_df

        shift_subscripts = []
        shift_values = []
        grouping_subscripts = []
        for column in primary_variable.base_df.columns:
            if column in shifts_by_subscript_name:
                shift = shifts_by_subscript_name[column]
            else:
                shift = 0

            if shift == 0:
                grouping_subscripts.append(column)
            else:
                shift_subscripts.append(column)
                shift_values.append(shift)

        primary_base_df = primary_variable.base_df.copy()

        if len(grouping_subscripts) > 0:
            grouped_df = primary_base_df.groupby(grouping_subscripts)
        else:
            grouped_df = primary_base_df

        for column, shift in zip(shift_subscripts, shift_values):
            shifted_column = grouped_df[column].shift(shift).reset_index(drop=True)
            primary_base_df[column] = shifted_column

        target_base_df = target_variable.base_df.copy()
        target_base_df.columns = list(target_variable_subscripts)

        target_base_df["__in_dataframe_index"] = pandas.Series(range(target_base_df.shape[0]), dtype = pandas.Int64Dtype())

        key_name = f"subscript__{self.get_unique_number()}"

        self.generated_subscript_dict[key_name] = (
            primary_base_df
            .merge(target_base_df, on=target_variable_subscripts, how="left")
            ["__in_dataframe_index"]
            # NAs correspond to out of bounds accesses -- those should map 
            # to zero (and any parameter that needs to do out of bounds
            # accesses will have zeros allocated for the last element)
            .fillna(target_base_df.shape[0])
            .astype(int)
            .to_numpy()
        )

        # If this is a recursively assigned parameter and there are groupings then we'll
        # need to generate some special values for the jax.lax.scan recursive assignment
        # implementation
        if (
            len(grouping_subscripts) > 0 and
            (primary_variable_name == target_variable_name) and
            (target_variable.variable_type == VariableType.ASSIGNED_PARAM)
        ):
            self.first_in_group_indicator[primary_variable_name] = (~primary_base_df.duplicated(subset=grouping_subscripts)).to_numpy()

        return key_name

    def __str__(self):
        return pprint.pformat(self.symbol_dict)
