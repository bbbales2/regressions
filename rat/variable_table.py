from collections import namedtuple
from typing import Any, Dict, List, Tuple, Iterable, Union, TypeVar, NamedTuple

import pandas
from sortedcontainers import SortedDict, SortedSet

K = TypeVar("K")
V = TypeVar("V")


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
    read_arguments_set: SortedSet
    write_arguments_set: SortedSet

    def __init__(self, name: str, argument_count: int):
        super().__init__(name, argument_count)
        self.read_arguments_set = SortedSet()
        self.write_arguments_set = SortedSet()

    def __len__(self):
        return len(self.read_arguments_set)

    def __iter__(self):
        for args in self.read_arguments_set:
            yield args

    def __call__(self, *args) -> V:
        self.write_arguments_set.add(args)
        return None

    def itertuples(self) -> Iterable[NamedTuple]:
        Type = namedtuple(f"{self.name}_domain", self.subscripts)
        for arguments in self.read_arguments_set:
            yield Type(*arguments)

    def swap_and_clear_write_buffer(self):
        self.read_arguments_set = self.write_arguments_set
        self.write_arguments_set = SortedSet()

    def buffers_are_different(self) -> bool:
        read_size = len(self.read_arguments_set)
        write_size = len(self.write_arguments_set)
        intersection_size = len(self.read_arguments_set.intersection(self.write_arguments_set))
        return read_size != intersection_size or write_size != intersection_size

    def get_index(self, args):
        return self.read_arguments_set.index(args)

    def get_value(self, args):
        raise TypeError("Internal error: cannot call get_value on a dynamic record")


class AssignedVariableRecord(DynamicVariableRecord):
    def __repr__(self) -> str:
        return f"{self.name}[{','.join(self.subscripts)}]@transformed"


class SampledVariableRecord(DynamicVariableRecord):
    def __repr__(self) -> str:
        return f"{self.name}[{','.join(self.subscripts)}]@sampled"


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
            data_name = get_dataframe_name_by_column_name(self.name, data)
        except KeyError:
            raise KeyError(f"{self.name} not found in the input data")

        df = data[data_name]

        if data_subscripts == ():
            raise Exception("Unimplemented")
        else:
            if data_subscripts is None:
                raise KeyError(f"No subscripts to bind")

            for subscript in data_subscripts:
                if subscript not in df.columns:
                    raise KeyError(f"{self.name} found in {data_name}, but subscript {subscript} not found")

        for row in df.itertuples():
            row_as_dict = row._asdict()
            arguments = tuple(row_as_dict[subscript] for subscript in data_subscripts)
            return_value = row_as_dict[self.name]

            if arguments in self.values:
                existing_value = self.values[arguments]
                if existing_value != return_value:
                    arguments_string = (
                        ",".join(f"{subscript} = {arg}" for subscript, arg in zip(data_subscripts, arguments))
                    )
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

    def itertuples(self) -> Iterable[NamedTuple]:
        Type = namedtuple(f"{self.name}_value", (self.name, *self.subscripts))
        for arguments, value in self.values.items():
            yield Type(value, *arguments)

    def __repr__(self) -> str:
        return f"{self.name}[{','.join(self.subscripts)}]@data"


class VariableTable:
    variable_dict: SortedDict[str, VariableRecord]

    def __init__(self):
        self.variable_dict = SortedDict()

    def __setitem__(self, variable_name: str, record: VariableRecord):
        self.variable_dict[variable_name] = record

    def __contains__(self, name: str) -> bool:
        return name in self.variable_dict

    def __getitem__(self, variable_name: str) -> VariableRecord:
        return self.variable_dict[variable_name]

    def __iter__(self) -> Iterable[str]:
        for name in self.variable_dict:
            yield name

    def variables(self):
        for variable in self.variable_dict.values():
            yield variable

    @property
    def unconstrained_parameter_size(self) -> int:
        size = 0

        for variable_name, record in self.variable_dict.items():
            if isinstance(record, SampledVariableRecord):
                size += len(record)

        return size
