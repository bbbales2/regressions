from dataclasses import dataclass
import jax.numpy as jnp
import pandas
from typing import List, Dict, Tuple, Union


class Index:
    # base_df is the dataframe of the actual parameters
    base_df: pandas.DataFrame
    # df has been extended to support shifts -- indices in
    # here that aren't in base_df correspond to zeros (not sampled)
    df: pandas.DataFrame
    levels: List
    indices: Dict
    # Maintain a list of all the shifts used to access a variable
    shifts_list: List[Tuple[Union[str, None]]]

    def __init__(self, unprocessed_df: pandas.DataFrame):
        # Rows of unprocessed_df are considered to be indexes into
        # another variable.
        #
        # Every column of unprocessed_df is considered a different
        # dimension.
        columns = unprocessed_df.columns

        base_df = unprocessed_df.drop_duplicates().sort_values(list(columns)).reset_index(drop=True)

        for column, dtype in zip(base_df.columns, base_df.dtypes):
            if pandas.api.types.is_integer_dtype(dtype):
                base_df[column] = base_df[column].astype(pandas.Int64Dtype())

        self.base_df = base_df
        self.df = self.base_df
        self.shifts_list = []

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
        self.df["__index"] = range(len(self.df.index))

        self.levels = [row for row in self.df.itertuples(index=False)]
        self.indices = {}
        for i, level in enumerate(self.levels):
            self.indices[level] = i

    def get_numpy_indices(self, df):
        df = df.copy()
        df.columns = self.base_df.columns
        print("------")
        print("self.base_df")
        print(self.base_df)
        print("self.df:")
        print(self.df)
        print("df:")
        print(df)
        print("merged:")
        print(df.merge(self.df, on=list(self.base_df.columns), how="left", validate="many_to_one",)[
            "__index"
        ])
        df.columns = self.base_df.columns
        return (df.merge(self.df, on=list(self.base_df.columns), how="left", validate="many_to_one",))[
            "__index"
        ].to_numpy(dtype=int)


@dataclass
class Data:
    name: str
    series: pandas.Series

    def to_numpy(self):
        return jnp.array(self.series).reshape((len(self.series), 1))

    def code(self):
        return f"data__{self.name}"


@dataclass
class Param:
    name: str
    index: Index = None
    lower: float = float("-inf")
    upper: float = float("inf")

    def set_constraints(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def scalar(self):
        if self.index:
            return False
        return True

    def size(self):
        if self.scalar():
            return None
        else:
            return len(self.index.base_df.index)

    def padded_size(self):
        if self.scalar():
            return None
        else:
            return len(self.index.df.index)

    def code(self):
        return f"param__{self.name}"


@dataclass
class AssignedParam:
    name: str
    rhs: None
    index: Index = None

    def size(self):
        if not self.index:
            return None
        else:
            return len(self.index.base_df.index)

    def code(self):
        return f"assigned_param__{self.name}"


@dataclass
class IndexUse:
    names: Tuple[str]
    df: pandas.DataFrame
    index: Index
    shifts: Tuple[Union[str, None]] = None

    def to_numpy(self):
        self.df = self.df.loc[:, self.names]
        shifted_df = self.index.compute_shifted_df(self.df, self.shifts)
        indices = self.index.get_numpy_indices(shifted_df)
        return jnp.array(indices, dtype=int).reshape((indices.shape[0], 1))

    def code(self):
        if all(shift is None for shift in self.shifts):
            return f"index__{'_'.join(self.names)}"
        else:
            return f"index__{'_'.join(self.names)}__{'_'.join([str(shift) for shift in self.shifts])}"
