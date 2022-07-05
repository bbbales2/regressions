from dataclasses import dataclass, field, replace
from enum import Enum
from multiprocessing.sharedctypes import Value
import numpy
import pandas
import pprint
from typing import Dict, Set, Tuple, Iterable, Union

from .exceptions import CompileError, MergeError


class VariableType(Enum):
    DATA = 1
    PARAM = 2
    ASSIGNED_PARAM = 3


class VariableTracer:
    args_set: Set[Tuple]

    def __init__(self):
        self.args_set = set()

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            return 0.0

        if len(kwargs) > 0:
            raise ValueError("Internal compiler error: A VariableTracer should not be traced with named arguments")
        self.args_set.add(args)
        # TODO: This is here so that code generation can work without worrying about types
        return 0.0

    def __len__(self):
        return len(self.args_set)

    def __iter__(self):
        for args in self.args_set:
            yield args

    def __repr__(self):
        return f"T{str(self.args_set)}"


class DataTracer(VariableTracer):
    def __init__(self, df: pandas.DataFrame):
        self.df = df
        self.args_set = set()
        for args in self.df.itertuples(index=False, name=None):
            self.args_set.add(args)

    def __call__(self, output_column_name, **kwargs):
        row_selector = numpy.logical_and.reduce([self.df[column] == value for column, value in kwargs.items()])

        row_df = self.df[row_selector]

        if len(row_df) == 0:
            raise ValueError(f"Internal compiler error: Data not defined for {kwargs}")
        elif len(row_df) > 1:
            raise ValueError(f"Internal compiler error: Data is not defined uniquely for {kwargs}")
        # TODO: This is here so that code generation can work without worrying about types
        return row_df.iloc[0][output_column_name]


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

    subscripts_requested_as_data: These are subscripts that should be generated as data on request
    """

    name: str
    variable_type: VariableType
    base_df: pandas.DataFrame

    renamed: bool = False
    constraints_set: bool = False

    constraint_lower: float = field(default=float("-inf"))
    constraint_upper: float = field(default=float("inf"))
    pad_needed: bool = field(default=False)

    unconstrained_vector_start_index: int = field(default=None, init=False)
    unconstrained_vector_end_index: int = field(default=None, init=False)

    subscripts_requested_as_data: set = field(default_factory=lambda: set())

    _lookup_cache: dict = field(default_factory=dict)

    @property
    def subscripts(self):
        if self.base_df is not None:
            return tuple(column for column in self.base_df.columns if column != self.name)
        else:
            return ()

    def rename(self, subscript_names: Tuple[str]):
        if self.renamed:
            raise AttributeError("Internal compiler error: A variable cannot be renamed twice")

        if self.base_df is not None:
            if len(self.base_df) > 0:
                raise ValueError("Internal compiler error: Renaming must be done before the dataframe is set")

            if len(self.base_df.columns) != len(subscript_names):
                raise ValueError("Internal compiler error: Number of subscript names must match number of dataframe columns")

        self.base_df = pandas.DataFrame(columns=subscript_names)
        self.renamed = True

    def suggest_names(self, subscript_names: Tuple[str]):
        """
        Set subscript names.

        This function should only be called once, it it is called more
        than that it will raise an AttributeError

        This function will raise a ValueError if called with the wrong
        number of names
        """
        if not self.renamed:
            if self.base_df is not None:
                if len(self.base_df.columns) != len(subscript_names):
                    raise AttributeError("Internal compiler error: Number of subscript names must match number of dataframe columns")
                if (self.base_df.columns != subscript_names).any():
                    raise AttributeError("Internal compiler error: If variable isn't renamed, then all subscript references must match")
            self.base_df = pandas.DataFrame(columns=subscript_names)

    def get_tracer(self):
        """
        Generate a tracer
        """
        if self.variable_type == VariableType.DATA:
            return DataTracer(self.base_df)
        else:
            return VariableTracer()

    def ingest_new_trace(self, tracer: VariableTracer):
        """
        Rebuild the base_df based on a tracer object. Return true If there are any differences between the
        new and old dataframes.
        """
        if len(tracer) == 0:
            if self.base_df is not None:
                self.base_df = pandas.DataFrame(columns=self.base_df.columns)
                return True
            else:
                return False
        else:
            new_base_df = pandas.DataFrame(tracer, columns=self.base_df.columns)
            new_len = len(new_base_df)

            if self.base_df is None:
                previous_len = 0
                inner_join_len = 0
            else:
                previous_len = len(self.base_df)
                inner_join_len = len(self.base_df.merge(new_base_df, on=tuple(self.base_df.columns), how="inner"))

            self.base_df = new_base_df.sort_values(list(new_base_df.columns)).reset_index(drop=True)

            self._lookup_cache = {}
            for row in self.base_df.itertuples(index=True, name=None):
                self._lookup_cache[row[1:]] = row[0]

            return (previous_len != inner_join_len) | (new_len != inner_join_len)

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

    def subscript_as_data_name(self, column: str):
        return f"{self.name}__{column}"

    def request_subscript_as_data(self, column: str):
        if column not in self.subscripts:
            raise ValueError(f"Internal compiler error: {column} not found among subscripts")
        self.subscripts_requested_as_data.add(column)
        return self.subscript_as_data_name(column)

    def get_numpy_names(self) -> Iterable[str]:
        """
        Get names of to-be-materialized data variables
        """
        for subscript in self.subscripts:
            yield self.subscript_as_data_name(subscript)

    def get_numpy_arrays(self) -> Iterable[Tuple[str, numpy.ndarray]]:
        """
        Materialize data variable as numpy arrays
        """

        def replace_int_types(array: numpy.ndarray) -> numpy.ndarray:
            # Pandas Int64Dtype()s don't play well with jax
            if series.dtype == pandas.Int64Dtype():
                return series.astype(int).to_numpy()
            else:
                return series.to_numpy()

        if self.variable_type == VariableType.DATA:
            series = self.base_df[self.name]
            yield self.name, replace_int_types(series)

        for column in self.subscripts:
            series = self.base_df[column]
            yield self.subscript_as_data_name(column), replace_int_types(series)

    def lookup(self, *args):
        if args in self._lookup_cache:
            return self._lookup_cache[args]
        else:
            return -1


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

    def get_dataframe_name(self, variable_name):
        """
        Look up the dataframe associated with a given variable name

        Exactly one dataframe must be found or a KeyError will be thrown
        """
        matching_dfs = []
        for key, data_df in self.data.items():
            if variable_name in data_df:
                matching_dfs.append(key)

        if len(matching_dfs) == 0:
            raise KeyError(f"Variable {variable_name} not found")
        elif len(matching_dfs) > 1:
            raise KeyError(f"Variable {variable_name} is ambiguous; it is found in dataframes {matching_dfs}")

        return matching_dfs[0]

    def insert(
        self,
        variable_name: str,
        variable_type: VariableType = None,
    ):
        """
        Insert a variable record. If it is data, attach an input dataframe to it
        """
        if variable_type == VariableType.DATA:
            base_df = self.data[self.get_dataframe_name(variable_name)]
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

        # TODO: It would be nice if this logic weren't different for data and parameters
        if target_variable.variable_type == VariableType.DATA:
            # If the target variable is data, then assume the subscripts here are
            # referencing columns of the dataframe by name
            target_base_df = target_variable.base_df[list(target_variable_subscripts)].copy()
        else:
            # If the target variable is parameter, then assume the subscripts here are
            # referencing columns of the dataframe by position
            target_base_df = target_variable.base_df.copy()
            target_base_df.columns = list(target_variable_subscripts)

        target_base_df["__in_dataframe_index"] = pandas.Series(range(target_base_df.shape[0]), dtype=pandas.Int64Dtype())

        key_name = f"subscript__{self.get_unique_number()}"

        try:
            in_dataframe_index = primary_base_df.merge(target_base_df, on=target_variable_subscripts, how="left", validate="many_to_one")[
                "__in_dataframe_index"
            ]
        except pandas.errors.MergeError as e:
            base_msg = f"Unable to merge {target_variable_name} into {primary_variable_name} on subscripts {target_variable_subscripts}"
            extended_msg = (
                base_msg + f" because values of {target_variable_name} are not unique with given subscripts"
                if "many-to-one" in str(e)
                else ""
            )
            raise MergeError(extended_msg)

        if target_variable.variable_type == VariableType.DATA:
            # NAs in a data dataframe should not be ignored -- these
            # all need to be defined
            na_count = in_dataframe_index.isna().sum()
            if na_count > 0:
                for row in primary_base_df[in_dataframe_index.isna()].itertuples(index=False):
                    dataframe_name = self.get_dataframe_name(target_variable_name)
                    base_msg = (
                        f"Unable to merge {target_variable_name} into {primary_variable_name} on subscripts {target_variable_subscripts}"
                    )
                    extended_msg = (
                        base_msg + f" because there are {na_count} required values not found in {dataframe_name}"
                        f" (associated with {target_variable_name}). For instance, there should be a value for"
                        f" {row} but this is not there."
                    )
                    raise MergeError(extended_msg)

            filled_in_dataframe_index = in_dataframe_index
        else:
            filled_in_dataframe_index = (
                in_dataframe_index
                # NAs correspond to out of bounds accesses -- those should map
                # to zero (and any parameter that needs to do out of bounds
                # accesses will have zeros allocated for the last element)
                .fillna(target_base_df.shape[0])
            )

        self.generated_subscript_dict[key_name] = filled_in_dataframe_index.astype(int).to_numpy()

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
