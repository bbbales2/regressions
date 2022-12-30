from collections import namedtuple
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Iterable, Union, TypeVar, NamedTuple, Iterator, Optional

import jax
import numpy
import pandas
from sortedcontainers import SortedDict, SortedSet

K = TypeVar("K")
V = TypeVar("V")


def _get_dataframe_name_by_variable_and_column_name(
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


@dataclass
class VariableRecord:
    name: str
    argument_count: int

    _subscripts: Optional[Tuple[str]] = None
    renamed: bool = False
    mapping: SortedDict = field(default_factory=SortedDict)

    @property
    def subscripts(self):
        if self.renamed:
            return self._subscripts
        else:
            return tuple(f"arg{n}" for n in range(self.argument_count))

    def rename(self, subscript_names: Iterable[str]):
        if self.renamed:
            raise AttributeError("Internal compiler error: A variable cannot be renamed twice")

        self._subscripts = tuple(subscript_names)
        self.renamed = True

    def __iter__(self):
        for args in self.mapping:
            # TODO: I haven't been able to decide between consistently treating args as tuples
            #  or treating the length 1 tuples like scalars
            if len(args) == 1:
                yield args[0]
            else:
                yield args

    def __len__(self):
        return len(self.mapping)

    def add(self, *args):
        self.mapping[args] = None

    def get_index(self, *args):
        try:
            return self.mapping.index(args)
        except ValueError:
            raise KeyError(f"{self.name}[{','.join(str(arg) for arg in args)}] not found")

    def get_value(self, *args):
        try:
            return self.mapping[args]
        except KeyError:
            raise KeyError(f"{self.name}[{','.join(str(arg) for arg in args)}] not found")

    def opportunistic_dict_iterator(self) -> Iterator[Dict[str, Any]]:
        for keys, value in self.mapping.items():
            keys_with_names = dict(zip(self.subscripts, keys))
            if value is not None:
                yield {**keys_with_names, self.name: value}
            else:
                yield keys_with_names

    # TODO: This should probably be __str__
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}[{','.join(self.subscripts)}], {len(self.mapping)})"


class DynamicVariableRecord(VariableRecord):
    pass


class AssignedVariableRecord(DynamicVariableRecord):
    pass


class RecurrentVariableRecord(AssignedVariableRecord):
    pass


class SampledVariableRecord(DynamicVariableRecord):
    pass


class ConstantVariableRecord(VariableRecord):
    def bind(self, data_subscripts: List[str], data: Dict[str, pandas.DataFrame]):
        """
        Bind function to data
        """
        # Find if there's a dataframe that supplies the requested values
        try:
            data_name = _get_dataframe_name_by_variable_and_column_name(data, subscript_name=self.name)
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

            if arguments in self.mapping:
                existing_value = self.mapping[arguments]
                if existing_value != return_value:
                    arguments_string = ",".join(f"{subscript} = {arg}" for subscript, arg in zip(data_subscripts, arguments))
                    raise Exception(
                        f"Error binding {self.name} to dataframe {data_name}. Multiple rows matching"
                        f" {arguments_string} with different values ({existing_value}, {return_value})"
                    )

            self.mapping[arguments] = return_value


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

    def get_unconstrained_parameters(self, unconstrained_parameter_vector: jax.numpy.ndarray) -> Dict[str, jax.numpy.ndarray]:
        unconstrained_parameter_size = self.unconstrained_parameter_size
        expected_shape = (unconstrained_parameter_size,)

        if unconstrained_parameter_vector.shape != expected_shape:
            raise ValueError(f"Unconstrained variable must be of shape {expected_shape}, found {unconstrained_parameter_vector.shape}")

        used = 0
        parameters = {}
        for variable in self.variables():
            if not isinstance(variable, SampledVariableRecord):
                continue

            if len(variable.subscripts) > 0:
                size = len(variable)
                parameters[variable.name] = unconstrained_parameter_vector[used : used + size]
                used += size
            else:
                parameters[variable.name] = unconstrained_parameter_vector[used]
                used += 1
        return parameters

    @property
    def unconstrained_parameter_size(self) -> int:
        size = 0

        for variable_name, record in self.variable_components.items():
            if isinstance(record, SampledVariableRecord):
                size += len(record)

        return size
