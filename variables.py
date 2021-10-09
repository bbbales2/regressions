from dataclasses import dataclass
import numpy
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

    def init(self, scope):
        scope[self.code()] = self.series

    def code(self):
        return f"data__{self.name}"

@dataclass
class Param:
    name : str
    index : Index = None

    def initialize(self, scope):
        scope[self.code()] = numpy.zeros(len(self.index.levels))

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

    def code(self):
        return f"index__{'_'.join(self.names)}"

@dataclass
class ParamUse:
    param : Param
    index_use : IndexUse = None