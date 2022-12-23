from collections import namedtuple
from typing import Any, Dict, List, Tuple, Iterable, Union, TypeVar, NamedTuple, Iterator

import jax.numpy
import pandas
from sortedcontainers import SortedDict, SortedSet

K = TypeVar("K")
V = TypeVar("V")


def get_dataframe_name_by_variable_and_column_name(
    data: Dict[str, pandas.DataFrame], variable_name: str = None, subscript_name: str = None
):
    """
    Look up the dataframe associated with a given variable and/or subscript name. variable_name and column_name
    should both be within the same one and only one dataframe.

    Exactly one dataframe must be found or a KeyError will be thrown
    """
    matching_dfs = []
    for key, data_df in data.items():
        variable_found = subscript_found = False
        if variable_name in data_df:
            variable_found = True
        if subscript_name in data_df:
            subscript_found = True

        if (variable_name and variable_found) and (subscript_found and subscript_name):
            # If both variable and subscript names are given, both must exist
            matching_dfs.append(key)
        elif (variable_name and variable_found) or (subscript_name and subscript_found):
            # Just either one supplied means only check for that one
            matching_dfs.append(key)

    if len(matching_dfs) == 0:
        if subscript_name and variable_name:
            raise KeyError(f"Subscript '{subscript_name}' for data variable '{variable_name}' not found")
        elif subscript_name:
            raise KeyError(f"Subscript '{subscript_name}' not found")
        elif variable_name:
            raise KeyError(f"Data variable '{variable_name}' not found")

    elif len(matching_dfs) > 1:
        # If we have multiple dataframes containing a subscript, raise an error if the variable isn't a data column
        # and the variable is present in multiple dataframes.
        if subscript_name and variable_name:
            raise KeyError(
                f"Subscript '{subscript_name}' for data variable '{variable_name}' is ambiguous; it is found in dataframes {matching_dfs}"
            )
        elif variable_name:
            raise KeyError(f"Data variable '{variable_name}' is ambiguous; it is found in dataframes {matching_dfs}")
        elif subscript_name:
            raise KeyError(f"Subscript '{subscript_name}' is ambiguous; it is found in dataframes {matching_dfs}")

    return matching_dfs[0]


class VariableRecord:
    """
    A record within the VariableTable

    name: Name of the variable
    argument_count: Number of arguments
    """

    name: str
    argument_count: int

    _subscripts: Union[None, Tuple[str]]
    renamed: bool

    def __init__(self, name: str, argument_count: int):
        self.name = name
        self.argument_count = argument_count
        self._subscripts = None
        self.renamed = False

    @property
    def subscripts(self):
        if self._subscripts:
            return self._subscripts
        else:
            return tuple(f"arg{n}" for n in range(self.argument_count))

    def rename(self, subscript_names: Iterable[str]):
        """
        Set subscript names.

        This function should only be called once, if it is called more
        than that it will raise an AttributeError

        This function will raise a ValueError if called after tracing has
        been done (tracing uses the values here)
        """
        if self.renamed:
            raise AttributeError("Internal compiler error: A variable cannot be renamed twice")

        self._subscripts = tuple(subscript_names)
        self.renamed = True

    def suggest_names(self, subscript_names: List[str]):
        if not self.renamed:
            if self._subscripts is not None:
                if len(self._subscripts) != len(subscript_names):
                    raise AttributeError("Internal compiler error: Number of subscript names must match between uses")
                if self._subscripts != subscript_names:
                    raise AttributeError("Internal compiler error: If variable isn't renamed, then all uses must match")
            self._subscripts = tuple(subscript_names)

    def __len__(self):
        raise Exception("Internal error: Use subclass instead")

    def get_index(self, args):
        raise Exception("Internal error: Use a subclass instead")

    def get_value(self, args):
        raise Exception("Internal error: Use a subclass instead")

    def swap_and_clear_write_buffer(self):
        raise Exception("Internal error: Use a dynamic subclass instead")

    def buffers_are_different(self):
        raise Exception("Internal error: Use a dynamic subclass instead")

    def itertuples(self) -> Iterable[NamedTuple]:
        raise Exception("Internal error: Use a subclass instead")


class DynamicVariableRecord(VariableRecord):
    arguments_set: SortedSet

    def __init__(self, name: str, argument_count: int):
        super().__init__(name, argument_count)
        self.arguments_set = SortedSet()

    def __len__(self):
        return len(self.arguments_set)

    def __iter__(self):
        for args in self.arguments_set:
            yield args

    def __call__(self, *args) -> V:
        self.arguments_set.add(args)
        return None

    def itertuples(self) -> Iterable[NamedTuple]:
        Type = namedtuple(f"{self.name}_domain", self.subscripts)
        for arguments in self.arguments_set:
            yield Type(*arguments)

    def get_index(self, args):
        return self.arguments_set.index(args)

    def get_value(self, args):
        raise TypeError("Internal error: cannot call get_value on a dynamic record")


class AssignedVariableRecord(DynamicVariableRecord):
    def __repr__(self) -> str:
        return f"{self.name}[{','.join(self.subscripts)}]@transformed"


class SampledVariableRecord(DynamicVariableRecord):
    def __repr__(self) -> str:
        return f"{self.name}[{','.join(self.subscripts)}]@sampled"

    def as_assigned_variable_record(self) -> AssignedVariableRecord:
        assigned_variable_record = AssignedVariableRecord(self.name, self.argument_count)
        if self.renamed:
            assigned_variable_record.rename(self.subscripts)
        assigned_variable_record.arguments_set = self.arguments_set
        return assigned_variable_record


class ConstantVariableRecord(VariableRecord):
    values: SortedDict
    value_type: Any

    def __init__(self, name: str, argument_count: int):
        super().__init__(name, argument_count)
        self.values = SortedDict()
        self.value_type = None

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterable[K]:
        for args in self.values:
            yield args

    def __call__(self, *args: K):
        try:
            return self.values[args]
        except KeyError:
            raise KeyError(f"Argument {args} not found in values placeholder")

    def bind(self, data_subscripts: List[str], data: Dict[str, pandas.DataFrame]):
        """
        Bind function to data
        """
        # Find if there's a dataframe that supplies the requested values
        try:
            data_name = get_dataframe_name_by_variable_and_column_name(data, subscript_name=self.name)
        except KeyError as e:
            raise KeyError(f"{self.name} not found in the input data") from e

        df = data[data_name]
        if data_subscripts == ():
            raise Exception("Unimplemented")
        else:
            if data_subscripts is None:
                raise Exception(f"No subscripts to bind but {self.name} found in {data_name} input data")

            for subscript in data_subscripts:
                if subscript not in df.columns:
                    raise Exception(f"{self.name} found in {data_name}, but subscript {subscript} not found")

        # Iterate over each row of the target dataframe
        row: NamedTuple
        for row in df.itertuples():  # row here is a namedtuple
            row_as_dict = row._asdict()
            # Pluck out only the relevant columns
            arguments = tuple(row_as_dict[subscript] for subscript in data_subscripts)
            return_value = row_as_dict[self.name]

            if arguments in self.values:
                existing_value = self.values[arguments]
                if existing_value != return_value:
                    arguments_string = ",".join(f"{subscript} = {arg}" for subscript, arg in zip(data_subscripts, arguments))
                    raise Exception(
                        f"Error binding {self.name} to dataframe {data_name}. Multiple rows matching"
                        f" {arguments_string} with different values ({existing_value}, {return_value})"
                    )

            if self.value_type is None:
                self.value_type = type(return_value)
            else:
                if not isinstance(return_value, self.value_type):
                    raise Exception("Value type inconsistent")

            self.values[arguments] = return_value

    def get_index(self, args):
        return self.values.index(args)

    def get_value(self, args):
        return self.values[args]

    def itertuples(self) -> Iterator[NamedTuple]:
        Type = namedtuple(f"{self.name}_value", (self.name, *self.subscripts))
        for arguments, value in self.values.items():
            yield Type(value, *arguments)

    def __repr__(self) -> str:
        return f"{self.name}[{','.join(self.subscripts)}]@data"


class VariableTable:
    variable_components: SortedDict[str, VariableRecord]

    def __init__(self):
        self.variable_components = SortedDict()

    def __getitem__(self, variable_name: str) -> VariableRecord:
        return self.variable_components[variable_name]

    def __setitem__(self, variable_name: str, record: VariableRecord):
        self.variable_components[variable_name] = record

    def __contains__(self, name: str) -> bool:
        return name in self.variable_components

    def __iter__(self) -> Iterator[str]:
        for name in self.variable_components:
            yield name

    def variables(self):
        for variable in self.variable_components.values():
            yield variable

    @property
    def unconstrained_parameter_size(self) -> int:
        size = 0

        for variable_name, record in self.variable_components.items():
            if isinstance(record, SampledVariableRecord):
                size += len(record)

        return size

    def transform(self, x: jax.numpy.ndarray) -> float:
        return sum(jax.numpy.sum(variable.evaluate()) for variable in self.variable_components)
