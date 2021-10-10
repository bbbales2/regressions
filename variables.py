from dataclasses import dataclass
import numpy
import jax.numpy as jnp
import pandas
from typing import List, Dict, Tuple

class Index:
    df : pandas.DataFrame
    levels : List
    indices : Dict

    def __init__(self, unprocessed_df : pandas.DataFrame):
        columns = unprocessed_df.columns

        self.df = (
            unprocessed_df
            .drop_duplicates()
            .sort_values(list(columns))
            .reset_index(drop = True)
        )

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

    def scalar(self):
        return self.index is None

    def size(self):
        if self.scalar():
            return None
        else:
            return len(self.index.levels)

    def code(self):
        return f"param__{self.name}"

@dataclass
class DataUse:
    data : Data

@dataclass
class IndexUse:
    names : Tuple[str]
    df : pandas.DataFrame
    index : Index

    def to_numpy(self):
        indices = []
        for row in self.df.itertuples(index = False):
            indices.append(self.index.get_index(row))
        return jnp.array(indices, dtype = int).reshape((len(self.df.index), 1))

    def code(self):
        return f"index__{'_'.join(self.names)}"

@dataclass
class ParamUse:
    param : Param
    index_use : IndexUse = None