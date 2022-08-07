from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from sortedcontainers import SortedDict, SortedSet
import numpy
import pandas
import pprint
from typing import Any, Dict, Set, Tuple, Iterable, Union, TypeVar, Generic, NamedTuple

from .exceptions import CompileError, MergeError


class VariableType(Enum):
    DATA = 1
    PARAM = 2
    ASSIGNED_PARAM = 3


class Tracer:
    args_set: SortedSet[Tuple]
    arg_types: Tuple

    def __init__(self, args_set=None, arg_types=None):
        self.args_set = args_set if args_set else SortedSet()
        self.arg_types = arg_types

    def __len__(self):
        return len(self.args_set)

    def __iter__(self):
        for args in self.args_set:
            yield args

    def __repr__(self):
        return f"F[{str(self.arg_types)} -> ?]"

    def argument_length(self):
        if self.arg_types:
            return len(self.arg_types)
        else:
            raise Exception("Argument length called before argument length known")

    def __call__(self, *args):
        self.validate_argument_types(args)
        self.args_set.add(args)

    def validate_argument_types(self, args):
        arg_types = tuple(type(arg) for arg in args)
        if self.arg_types == None:
            self.arg_types = arg_types
        else:
            if len(args) != len(self.arg_types):
                raise Exception("Inconsistent number of arguments")

            if arg_types != self.arg_types:
                raise Exception("Argument types inconsistent")

    def lookup(self, args) -> int:
        return self.args_set.index(args)

    def copy(self):
        return Tracer(self.args_set.copy(), self.arg_types)


K = TypeVar("K")  # TODO: Is this a useful way of typing this?
V = TypeVar("V")


class ValueTracer(Tracer, Generic[K, V]):
    values: SortedDict[K, V]
    arg_types: Tuple
    value_type: Any

    def __init__(self):
        self.values = SortedDict()
        self.arg_types = None
        self.value_type = None

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterable[K]:
        for args in self.values:
            yield args

    def __repr__(self) -> str:
        return f"F[{str(self.arg_types)} -> {str(self.value_type)}]"

    def __call__(self, *args: K):
        try:
            return self.values[args]
        except KeyError:
            raise Exception("Argument {args} not found in data tracer")

    def validate_return_type(self, value):
        value_type = type(value)
        if self.value_type == None:
            self.value_type = value_type
        else:
            if value_type != self.value_type:
                raise Exception("Value type inconsistent")

    def add(self, return_value: V, args: K):
        self.validate_argument_types(args)
        self.validate_return_type(return_value)
        self.values[args] = return_value

    def lookup(self, args: K) -> int:
        idx = self.values.index(args)
        return idx

    # TODO: This is really confusing lol
    def copy(self):
        return self


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
    argument_count: int
    variable_type: VariableType
    tracer: Tracer = None

    _subscripts: Tuple[str] = None
    renamed: bool = False
    constraints_set: bool = False

    constraint_lower: float = field(default=float("-inf"))
    constraint_upper: float = field(default=float("inf"))
    pad_needed: bool = field(default=False)

    unconstrained_vector_start_index: int = field(default=None, init=False)
    unconstrained_vector_end_index: int = field(default=None, init=False)

    subscripts_requested_as_data: set = field(default_factory=set)

    _lookup_cache: dict = field(default_factory=dict)

    @property
    def subscripts(self):
        if self._subscripts:
            return self._subscripts
        else:
            return tuple(f"arg{n}" for n in range(self.argument_count))

    def rename(self, subscript_names: Tuple[str]):
        """
        Set subscript names.

        This function should only be called once, it it is called more
        than that it will raise an AttributeError

        This function will raise a ValueError if called after tracing has
        been done (tracing uses the values here)
        """
        if self.renamed:
            raise AttributeError("Internal compiler error: A variable cannot be renamed twice")

        if self.tracer is not None:
            if len(self.tracer) > 0:
                raise ValueError("Internal compiler error: Renaming must be done before the dataframe is set")

        self._subscripts = subscript_names
        self.renamed = True

    def suggest_names(self, subscript_names: Tuple[str]):
        if not self.renamed:
            if self._subscripts is not None:
                if len(self._subscripts) != len(subscript_names):
                    raise AttributeError("Internal compiler error: Number of subscript names must match between uses")
                if self._subscripts != subscript_names:
                    raise AttributeError("Internal compiler error: If variable isn't renamed, then all uses must match")
            self._subscripts = subscript_names

    def bind(self, data_subscripts: Tuple[str], data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]):
        """
        Bind function to data
        """
        data_dict: Dict[str, pandas.DataFrame] = {}

        match data:
            case pandas.DataFrame():
                data_dict["__default"] = data.reset_index().copy()
            case dict():
                for key, value in data.items():
                    data_dict[key] = value.reset_index().copy()
            case _:
                raise Exception("Data must either be pandas data frames or a dict of pandas dataframes")

        # Convert all integer columns to a type that supports NA
        for key, data_df in data_dict.items():
            for column in data_df.columns:
                if pandas.api.types.is_integer_dtype(data_df[column].dtype):
                    data_df[column] = data_df[column].astype(pandas.Int64Dtype())

        # Find if there's a dataframe that supplies the requested values
        try:
            data_name = get_dataframe_name_by_column_name(self.name, data_dict)
        except KeyError as e:
            return

        # If we find something to bind to, do it and call this variable data
        self.variable_type = VariableType.DATA

        data_df = data_dict[data_name]

        if data_subscripts == ():
            raise Exception("Unimplemented")
        else:
            for subscript in data_subscripts:
                if subscript not in data_df.columns:
                    raise Exception(f"{self.name} found in {data_name}, but subscript {subscript} not found")

        if not self.tracer:
            self.tracer = ValueTracer()

        for row in data_df.itertuples():
            row_as_dict = row._asdict()
            arguments = tuple(row_as_dict[subscript] for subscript in data_subscripts)
            return_value = row_as_dict[self.name]

            if arguments in self.tracer:
                existing_value = self.tracer(*arguments)
                if existing_value != return_value:
                    arguments_string = ",".join(f"{subscript} = {arg}" for subscript, arg in zip(data_subscripts, arguments))
                    raise Exception(
                        f"Error binding {self.name} to dataframe {data_name}. Multiple rows matching"
                        f" {arguments_string} with different values ({existing_value}, {return_value})"
                    )

            self.tracer.add(return_value, arguments)

    def itertuple_type(self):
        match self.tracer:
            case ValueTracer():
                Value = namedtuple(f"{self.name}_value", (self.name, *self.subscripts))
                return Value
            case Tracer():
                Domain = namedtuple(f"{self.name}_domain", self.subscripts)
                return Domain

    def itertuples(self) -> Iterable[NamedTuple]:
        Type = self.itertuple_type()
        match self.tracer:
            case ValueTracer():
                for arg in self.tracer:
                    yield Type(self.tracer(*arg), *arg)
            case Tracer():
                for arg in self.tracer:
                    yield Type(*arg)

    def ingest_new_trace(self, tracer: Tracer) -> bool:
        """
        Ingest a new tracer. Return true If there are any differences between the new and old dataframes.
        """
        # See if new tracer is the same size
        found_difference = len(tracer) != len(self.tracer)
        # See if new tracer has exactly the same domain
        found_difference |= any(arg not in tracer for arg in self.tracer)

        self.tracer = tracer

        return found_difference

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

    def get_numpy_names(self, names = None) -> Iterable[str]:
        """
        Get names of to-be-materialized data variables
        """
        fields = self.itertuple_type()._fields
        if names is None:
            names = fields
        for field in fields:
            if field in names:
                yield self.subscript_as_data_name(field)

    def get_numpy_arrays(self, names = None) -> Iterable[Tuple[str, numpy.ndarray]]:
        """
        Materialize variable as numpy arrays
        """
        fields = self.itertuple_type()._fields
        if names is None:
            names = fields
        for field, values in zip(fields, zip(*self.itertuples())):
            if field in names:
                yield self.subscript_as_data_name(field), numpy.array(values)

    def lookup(self, args):
        return self.tracer.lookup(args)


def get_dataframe_name_by_column_name(column_name: str, data: Dict[str, pandas.DataFrame]):
    """
    Look up the dataframe associated with a given variable name

    Exactly one dataframe must be found or a KeyError will be thrown
    """
    matching_dfs = []
    for key, data_df in data.items():
        if column_name in data_df:
            matching_dfs.append(key)

    if len(matching_dfs) == 0:
        raise KeyError(f"Variable {column_name} not found")
    elif len(matching_dfs) > 1:
        raise Exception(f"Variable {column_name} is ambiguous; it is found in dataframes {matching_dfs}")

    return matching_dfs[0]


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

        self.generated_subscript_dict = {}
        self.first_in_group_indicator = {}

        self._unique_number = 0

    def get_unique_number(self):
        self._unique_number += 1
        return self._unique_number

    def tracers(self) -> Iterable[Tracer]:
        return {name: variable.tracer for name, variable in self.variable_dict.items()}

    def insert(self, variable_name: str, argument_count=0, variable_type: VariableType = None):
        """
        Insert a variable record. If it is data, attach an input dataframe to it
        """

        # Insert if new
        self.variable_dict[variable_name] = VariableRecord(
            variable_name,
            argument_count,
            variable_type,
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
                    nrows = len(record.tracer)
                else:
                    nrows = 1

                record.unconstrained_vector_start_index = current_index
                record.unconstrained_vector_end_index = current_index + nrows - 1
                current_index += nrows

        self.unconstrained_parameter_size = current_index
