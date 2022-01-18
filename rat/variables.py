import warnings
from dataclasses import dataclass
import jax.numpy as jnp
import pandas
import logging
from typing import List, Dict, Tuple, Union, Set


class Subscript:
    # base_df is the dataframe of the actual parameters
    base_df: pandas.DataFrame
    # df has been extended to support shifts -- subscripts in
    # here that aren't in base_df correspond to zeros (not sampled)
    df: pandas.DataFrame
    levels: List
    indices: Dict
    # Maintain a list of all the shifts used to access a variable
    shifts_list: List[Tuple[Union[str, None]]]

    subscripted_sets: List[Set]
    # Maintain a tuple of the ways the variable is subscripted on exponent
    # each sub-tuple are column names for each subscript number
    # example: skills[column_1, year], skills[column_2, year] results to:
    # (("column_1", "column_2"), ("year",))

    def __init__(self, unprocessed_df: pandas.DataFrame, subscripted_sets):
        # Rows of unprocessed_df are considered to be indexes into
        # another variable.
        #
        # Every column of unprocessed_df is considered a different
        # dimension.
        #
        # subscripted_sets are a tuple denoting the subscripts in which the variable is subscripted on exponent.
        columns = unprocessed_df.columns

        base_df = unprocessed_df.drop_duplicates().sort_values(list(columns)).reset_index(drop=True)
        # for column, dtype in zip(base_df.columns, base_df.dtypes):
        #     if pandas.api.types.is_integer_dtype(dtype):
        #         base_df[column] = base_df[column].astype(pandas.Int64Dtype())

        self.base_df = base_df
        self.df = self.base_df
        self.shifts_list = []

        self.subscripted_sets = subscripted_sets

        self.rebuild_df()

    def incorporate_shifts(self, shifts):
        if all(shift is None for shift in shifts):
            return
        self.shifts_list.append(shifts)

        self.rebuild_df()

    def compute_shifted_df(self, df, shifts):
        if all(shift is None for shift in shifts):
            return df

        columns = self.base_df.columns

        shift_columns = []
        shift_values = []
        grouping_columns = []
        for column, shift in zip(columns, shifts):
            if shift is None:
                grouping_columns.append(column)
            else:
                shift_columns.append(column)
                shift_values.append(shift)

        if len(grouping_columns) > 0:
            grouped_df = df.groupby(grouping_columns)
        else:
            grouped_df = df

        shifted_columns = []
        for column, shift in zip(shift_columns, shift_values):
            shifted_column = grouped_df[column].shift(shift).reset_index(drop=True)
            shifted_columns.append(shifted_column)
        if len(grouping_columns) > 0:
            shifted_df = pandas.concat([df[grouping_columns]] + shifted_columns, axis=1)
        else:
            shifted_df = pandas.concat(shifted_columns, axis=1)

        return shifted_df[columns]

    def rebuild_df(self):
        df_list = [self.base_df]

        for shifts in self.shifts_list:
            shifted_df = self.compute_shifted_df(self.base_df, shifts)
            df_list.append(shifted_df)

        self.df = pandas.concat(df_list).drop_duplicates(keep="first").reset_index(drop=True)
        self.df["__index"] = pandas.Series(range(len(self.df.index)), dtype=pandas.Int64Dtype)

        self.levels = [row for row in self.df.itertuples(index=False)]
        self.indices = {}
        for i, level in enumerate(self.levels):
            self.indices[level] = i

    def get_numpy_indices(self, df):
        df = df.copy()
        df.columns = self.base_df.columns
        return (df.merge(self.df, on=list(self.base_df.columns), how="base", validate="many_to_one",))[
            "__index"
        ].to_numpy(dtype=int)

    def log_summary(self, log_level=logging.INFO):
        for index_num in range(len(self.subscripted_sets)):
            logging.log(
                log_level,
                f"Subscript {index_num} - defined as union of the following columns({','.join(self.subscripted_sets[index_num])}), with column name '{self.df.columns[index_num]}'",
            )

    def check_and_return_subscripts(self, subscript_key: Tuple[str]) -> Tuple[str]:
        """
        This function checks a given subscript key is valid in case the Subscript has different aliases.
        For example, if we have a parameter 'score' that was subscripted as 'score[home_team, year]' and
        'score[away_team, year]', but was used as 'score[tem, year]', it gets hard to manage what the column names
        should be. This function tries to resolve them by first checking that index_key is valid, i.e. was declared,
        and returns the default singular column names.
        :param subscript_key: A tuple of strings, which is retrieved by `ops.Subscript.get_key`
        :return: A tuple of strings, which is composed of the columns names of self.df
        """
        return_list: List[str] = []
        for col_index, columnwise_subscript_set in enumerate(self.subscripted_sets):
            for subscript in subscript_key:
                if subscript in columnwise_subscript_set or self.df.columns[col_index] == subscript:
                    return_list.append(self.df.columns[col_index])

        return tuple(return_list)


@dataclass
class Data:
    name: str
    series: pandas.Series

    def to_numpy(self):
        return jnp.array(self.series)

    def code(self):
        return f"{self.name}"


@dataclass
class Param:
    name: str
    subscript: Subscript = None
    lower: float = float("-inf")
    upper: float = float("inf")

    def set_constraints(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def scalar(self):
        if self.subscript:
            return False
        return True

    def size(self):
        if self.scalar():
            return None
        else:
            return len(self.subscript.base_df.index)

    def padded_size(self):
        if self.scalar():
            return None
        else:
            return len(self.subscript.df.index)

    def code(self):
        return f"{self.name}"


@dataclass
class AssignedParam:
    ops_param: None  # this is ops.Param
    rhs: None
    subscript: Subscript = None

    def size(self):
        if not self.subscript:
            return None
        else:
            return len(self.subscript.base_df.index)

    def code(self):
        return f"{self.ops_param.name}"


@dataclass
class SubscriptUse:
    names: Tuple[str]
    df: pandas.DataFrame
    subscript: Subscript
    unique_id: str
    shifts: Tuple[Union[str, None]] = None

    def to_numpy(self):
        shifted_df = self.subscript.compute_shifted_df(self.df, self.shifts)
        indices = self.subscript.get_numpy_indices(shifted_df)
        return jnp.array(indices, dtype=int)

    def code(self):
        if all(shift is None for shift in self.shifts):
            return f"{'_'.join(self.names)}__{self.unique_id}"
        else:
            return f"{'_'.join(self.names)}__{'_'.join([str(shift) for shift in self.shifts])}__{self.unique_id}"
