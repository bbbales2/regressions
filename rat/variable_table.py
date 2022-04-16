from dataclasses import dataclass, field
from enum import Enum
import numpy
import pandas
import pprint
from typing import Dict, Set, Tuple, Iterable, Union

from .exceptions import CompileError, MergeError


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

    names_set: bool = False
    constraints_set: bool = False

    constraint_lower: float = field(default=float("-inf"))
    constraint_upper: float = field(default=float("inf"))
    pad_needed: bool = field(default=False)

    unconstrained_vector_start_index: int = field(default=None, init=False)
    unconstrained_vector_end_index: int = field(default=None, init=False)

    @property
    def subscripts(self):
        if self.base_df is not None:
            return tuple(self.base_df.columns)
        else:
            return ()

    def set_subscript_names(self, subscript_names: Tuple[str]):
        """
        Set subscript names.

        This function should only be called once, it it is called more
        than that it will raise an AttributeError

        This function will raise a ValueError if called with the wrong
        number of names
        """
        if not self.names_set:
            if len(self.base_df.columns) != len(subscript_names):
                raise ValueError("Internal compiler error: Number of subscript names must match number of dataframe columns")
            self.base_df.columns = subscript_names
            self.names_set = True
        else:
            raise AttributeError("Internal compiler error: If there are multiple renames, the rename values must match")

    def add_rows_from_dataframe(self, new_rows_df: pandas.DataFrame):
        """
        Add rows to the base_df, ensure there aren't any duplicates, and make
        sure that the base_df is sorted by it's columns (in order left to right)
        """
        if self.base_df is None:
            combined_df = new_rows_df
        else:
            combined_df = pandas.DataFrame(numpy.concatenate([self.base_df.values, new_rows_df.values]), columns=self.base_df.columns)

        self.base_df = combined_df.drop_duplicates().sort_values(list(combined_df.columns)).reset_index(drop=True)

    def set_constraints(self, constraint_lower: float, constraint_upper: float):
        """
        Set scalar constraints on a variable unless constraint_lower is negative
        infinity and constraint upper is positive infinity -- in that case do nothing!

        TODO: Should have the parser indicate no constraints in a different way so this
        function is less weird.

        Once this function is called, it can
        be called again but it will throw an AttributeError if on following calls
        the arguments do not exactly match what was passed on the first call.
        """
        if constraint_lower == float("-inf") and constraint_upper == float("inf"):
            return

        if not self.constraints_set:
            self.constraint_lower = constraint_lower
            self.constraint_upper = constraint_upper
            self.constraints_set = True
        else:
            if self.constraint_lower != constraint_lower or self.constraint_upper != constraint_upper:
                raise AttributeError("Internal compiler error: Once changed from defaults, constraints must match")

    def to_numpy(self) -> numpy.ndarray:
        """
        Materialize data variable as a numpy array
        """
        if self.variable_type == VariableType.DATA:
            series = self.base_df[self.name]
            # Pandas Int64Dtype()s don't play well with jax
            if series.dtype == pandas.Int64Dtype():
                return series.astype(int).to_numpy()
            else:
                return series.to_numpy()
        else:
            raise AttributeError("Internal compiler error: `to_numpy` can only be used if variable_type is data")


class VariableTable:
    variable_dict: Dict[str, VariableRecord]
    unconstrained_parameter_size: int
    data: Dict[str, pandas.DataFrame]

    generated_subscript_dict: Dict[str, numpy.ndarray]
    first_in_group_indicator: Dict[str, numpy.ndarray]

    _unique_number: int

    def __init__(self, data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]):
        self.variable_dict = {}
        self.data = {}
        self.unconstrained_parameter_size = None

        match data:
            case pandas.DataFrame():
                self.data["__default"] = data.copy()
            case _:
                for key, value in data.items():
                    self.data[key] = value.copy()

        # Convert all integer columns to a type that supports NA
        for key, data_df in self.data.items():
            for column in data_df.columns:
                if pandas.api.types.is_integer_dtype(data_df[column].dtype):
                    data_df[column] = data_df[column].astype(pandas.Int64Dtype())

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
        Insert a variable record. If it is data, attach an input dataframe to it

        Exactly one dataframe must be attached or a KeyError will be thrown
        """
        if variable_name not in self.variable_dict:
            if variable_type == VariableType.DATA:
                matching_dfs = []
                for key, data_df in self.data.items():
                    if variable_name in data_df:
                        matching_dfs.append(key)
                
                if len(matching_dfs) == 0:
                    raise KeyError(f"Variable {variable_name} not found")
                elif len(matching_dfs) > 1:
                    raise KeyError(f"Variable {variable_name} is ambiguous; it is found in dataframes {matching_dfs}")
  
                base_df = self.data[matching_dfs[0]]
            else:
                base_df = None

            # Insert if new
            self.variable_dict[variable_name] = VariableRecord(
                variable_name,
                variable_type,
                base_df,
            )

    def __contains__(self, name: str) -> bool:
        return name in self.variable_dict

    def __getitem__(self, variable_name: str) -> VariableRecord:
        return self.variable_dict[variable_name]

    def __iter__(self):
        for name in self.variable_dict:
            yield name

    def prepare_unconstrained_parameter_mapping(self):
        """
        Compute the size of vector that can hold everything and figure out
        how each record maps into this vector
        """
        current_index = 0

        for variable_name, record in self.variable_dict.items():
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

        shifts_by_subscript_name = {name: shift for name, shift in zip(target_variable_subscripts, shifts)}

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

        target_base_df = target_variable.base_df[list(target_variable_subscripts)].copy()

        target_base_df["__in_dataframe_index"] = pandas.Series(range(target_base_df.shape[0]), dtype=pandas.Int64Dtype())

        key_name = f"subscript__{self.get_unique_number()}"

        try:
            self.generated_subscript_dict[key_name] = (
                primary_base_df
                .merge(
                    target_base_df,
                    on=target_variable_subscripts,
                    how="left",
                    validate="many_to_one"
                )["__in_dataframe_index"]
                # NAs correspond to out of bounds accesses -- those should map
                # to zero (and any parameter that needs to do out of bounds
                # accesses will have zeros allocated for the last element)
                .fillna(target_base_df.shape[0])
                .astype(int)
                .to_numpy()
            )
        except pandas.errors.MergeError as e:
            base_msg = f"Unable to merge {target_variable_name} into {primary_variable_name} on subscripts {target_variable_subscripts}"
            extended_msg = base_msg + f" because values of {target_variable_name} are not unique with given subscripts" if "many-to-one" in str(e) else ""
            raise MergeError(extended_msg)

        # If this is a recursively assigned parameter and there are groupings then we'll
        # need to generate some special values for the jax.lax.scan recursive assignment
        # implementation
        if (
            len(grouping_subscripts) > 0
            and (primary_variable_name == target_variable_name)
            and (target_variable.variable_type == VariableType.ASSIGNED_PARAM)
        ):
            self.first_in_group_indicator[primary_variable_name] = (~primary_base_df.duplicated(subset=grouping_subscripts)).to_numpy()

        return key_name

    def __str__(self):
        return pprint.pformat(self.variable_dict)
