from dataclasses import dataclass, field
from enum import Enum
import numpy
import pandas
import pprint
from typing import Dict, Set, Tuple

from .exceptions import CompileError


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
    shifts: set of shift combinations used for this parameter. For example, param[shift(sub_1, 1), sub_2] would be saved
    as (1, 0)
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
    pad_needed: true if a padding is needed for the variable
    """

    name: str
    variable_type: VariableType
    subscripts: Set[Tuple[str]]  # tuple of (subscript_name, subscript_name, ...)
    # shifts: Set[Tuple[int]]
    subscript_length: int = field(default=0)
    subscript_alias: Tuple[str] = field(default=tuple())
    constraint_lower: float = field(default=float("-inf"))
    constraint_upper: float = field(default=float("inf"))
    base_df: pandas.DataFrame = field(default=None, init=False, repr=False)
    unconstrained_vector_start_index: int = field(default=None, init=False)
    unconstrained_vector_end_index: int = field(default=None, init=False)
    pad_needed: bool = field(default=False, init=False)


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
        self.first_in_group_indicator: Dict[str, numpy.ndarray] = {}

        self._unique_number = 0

    def get_unique_number(self):
        self._unique_number += 1
        return self._unique_number

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

    def lookup(self, variable_name: str) -> TableRecord:
        """
        Dictionary indexing
        """
        return self.symbol_dict[variable_name]

    def iter_records(self):
        for name, record in self.symbol_dict.items():
            yield name, record

    def build_base_dataframes(self, data_df: pandas.DataFrame):
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
                base_df = pandas.DataFrame()

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
                    base_df = pandas.concat([base_df, df]).drop_duplicates().sort_values(list(df.columns)).reset_index(drop=True)

                # For parameters, allocate space on the unconstrained parameter vector and record indices
                if record.variable_type == VariableType.PARAM:
                    nrows = base_df.shape[0]
                    # base_df["__index"] = pd.Series(range(current_index, current_index + nrows))
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

    def get_subscript_key(
        self,
        primary_variable_name: str,
        primary_subscript_names: Tuple[str],
        target_variable_name: str,
        target_subscript_names: Tuple[str],
        target_shift_amounts: Tuple[int],
    ):
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
                    subset_df.columns = (base_df_subscript_name,)
                    primary_base_df = pandas.merge(primary_base_df, subset_df, on=base_df_subscript_name, how="inner")

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
            shifted_column = grouped_df[column].shift(shift).reset_index(drop=True)
            primary_base_df[column] = shifted_column

        target_base_df = target_record.base_df.copy()
        target_base_df.columns = list(target_subscript_names)

        target_base_df["__in_dataframe_index"] = pandas.Series(range(target_base_df.shape[0]))

        key_name = f"subscript__{self.generated_subscript_count}"
        self.generated_subscript_count += 1
        output_df = pandas.merge(primary_base_df[list(target_subscript_names)], target_base_df, on=target_subscript_names, how="left")

        # number of columns being subscripted
        n_shifts = len(target_shift_amounts)
        if n_shifts > 1 and target_record.variable_type == VariableType.ASSIGNED_PARAM:
            self.first_in_group_indicator[target_variable_name] = (~output_df.duplicated(subset=grouping_subscripts)).to_numpy()

        output_df["__in_dataframe_index"] = output_df["__in_dataframe_index"].fillna(target_base_df.shape[0])
        self.generated_subscript_dict[key_name] = output_df["__in_dataframe_index"].to_numpy(dtype=int)

        return key_name

    def __str__(self):
        return pprint.pformat(self.symbol_dict)
