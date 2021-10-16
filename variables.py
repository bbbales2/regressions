from dataclasses import dataclass
import numpy
import jax.numpy as jnp
import pandas
from typing import List, Dict, Tuple

import ops

class Index:
    base_df : pandas.DataFrame
    df : pandas.DataFrame
    levels : List
    indices : Dict
    shift_columns_list : List[Tuple[str]] = []
    shift_list : List[int] = []

    def __init__(self, unprocessed_df : pandas.DataFrame):
        columns = unprocessed_df.columns

        self.base_df = (
            unprocessed_df
            .drop_duplicates()
            .sort_values(list(columns))
            .reset_index(drop = True)
        )

        self.df = self.base_df

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

        return pandas.concat([
            df.iloc[:, grouping_columns],
            df.groupby(grouping_columns).shift(shift).reset_index(drop = True)
        ], axis = 1).iloc[:, self.base_df.columns]

    def rebuild_df(self):
        df_list = [self.base_df]

        for shift_columns, shift in zip(self.shift_columns_list, self.shift_list):
            shifted_df = self.compute_shifted_df(self.base_df, shift_columns, shift)
            df_list.append(shifted_df)

        self.df = pandas.concat(df_list).drop_duplicates(keep = "first").reset_index(drop = True)

        self.levels = [row for row in self.df.itertuples(index = False)]
        self.indices = {}
        for i, level in enumerate(self.levels):
            self.indices[level] = i

    def get_level(self, i):
        return self.levels[i]

    def get_index(self, level):
        return self.indices[level]

@dataclass
class Data:
    name : str
    series : pandas.Series

    def to_numpy(self):
        return jnp.array(self.series).reshape((len(self.series), 1))

    def code(self):
        return f"data__{self.name}"

@dataclass
class Param:
    name : str
    index : Index = None
    lower : float = float("-inf")
    upper : float = float("inf")

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
    names : Tuple[str]
    df : pandas.DataFrame
    index : Index
    shift_columns : Tuple[str] = None
    shift : int = None

    def to_numpy(self):
        indices = []
        shifted_df = self.index.compute_shifted_df(self.df, self.shift_columns, self.shift)
        for row in shifted_df.itertuples(index = False):
            indices.append(self.index.get_index(row))
        return jnp.array(indices, dtype = int).reshape((len(self.df.index), 1))

    def code(self):
        return f"index__{'_'.join(self.names)}"
