from dataclasses import dataclass
import jax.numpy as jnp
import pandas
from typing import List, Dict, Tuple


class Index:
    # base_df is the dataframe of the actual parameters
    base_df: pandas.DataFrame
    # df has been extended to support shifts -- indices in
    # here that aren't in base_df correspond to zeros (not sampled)
    df: pandas.DataFrame
    levels: List
    indices: Dict
    # Maintain a list of all the shifts used to access a variable
    shift_columns_list: List[Tuple[str]]
    shift_list: List[int]

    def __init__(self, unprocessed_df: pandas.DataFrame):
        # Rows of unprocessed_df are considered to be indexes into
        # another variable.
        #
        # Every column of unprocessed_df is considered a different
        # dimension.
        columns = unprocessed_df.columns

        base_df = (
            unprocessed_df.drop_duplicates()
            .sort_values(list(columns))
            .reset_index(drop=True)
        )

        for column, dtype in zip(base_df.columns, base_df.dtypes):
            if pandas.api.types.is_integer_dtype(dtype):
                base_df[column] = base_df[column].astype(pandas.Int64Dtype())

        self.base_df = base_df
        self.df = self.base_df
        self.shift_columns_list = []
        self.shift_list = []

        self.rebuild_df()

    def incorporate_shifts(self, shift_columns, shift):
        if shift_columns is None:
            return

        self.shift_columns_list.append(shift_columns)
        self.shift_list.append(shift)

        self.rebuild_df()

    def compute_shifted_df(self, df, shift_columns, shift):
        if shift_columns is None:
            return df

        grouping_columns = list(set(self.base_df.columns) - set(shift_columns))

        if len(grouping_columns) > 0:
            grouped_df = df.groupby(grouping_columns)
            shifted_columns = []
            for column in shift_columns:
                shifted_column = grouped_df[column].shift(shift).reset_index(drop = True)
                shifted_columns.append(shifted_column)
            shifted_df = pandas.concat([df[grouping_columns]] + shifted_columns, axis=1)
            return shifted_df[list(self.base_df.columns)]
        else:
            return df.shift(shift)

    def rebuild_df(self):
        df_list = [self.base_df]

        for shift_columns, shift in zip(self.shift_columns_list, self.shift_list):
            shifted_df = self.compute_shifted_df(self.base_df, shift_columns, shift)
            df_list.append(shifted_df)

        self.df = (
            pandas.concat(df_list).drop_duplicates(keep="first").reset_index(drop=True)
        )
        self.df["__index"] = range(len(self.df.index))

        self.levels = [row for row in self.df.itertuples(index=False)]
        self.indices = {}
        for i, level in enumerate(self.levels):
            self.indices[level] = i

    def get_numpy_indices(self, df):
        df = df.copy()
        df.columns = self.base_df.columns
        return (
            df.merge(
                self.df,
                on=list(self.base_df.columns),
                how="left",
                validate="many_to_one",
            )
        )["__index"].to_numpy(dtype=int)


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
        return self.index is None

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
class IndexUse:
    names: Tuple[str]
    df: pandas.DataFrame
    index: Index
    shift_columns: Tuple[str] = None
    shift: int = None

    def to_numpy(self):
        shifted_df = self.index.compute_shifted_df(
            self.df, self.shift_columns, self.shift
        )
        indices = self.index.get_numpy_indices(shifted_df)
        return jnp.array(indices, dtype=int).reshape((indices.shape[0], 1))

    def code(self):
        if self.shift_columns is None:
            return f"index__{'_'.join(self.names)}"
        else:
            return f"index__{'_'.join(self.names)}__{'_'.join(self.shift_columns)}__{self.shift}"
