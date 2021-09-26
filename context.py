import pandas

class StanIndex:
    def __init__(self, df, values_df = None):
        self.columns = tuple(df.columns.to_list())
        
        self.source_df = df.copy()

        if values_df is not None:
            self.values_df = values_df.copy()
            self.values_df.columns = self.source_df.columns
        else:
            self.values_df = self.source_df

        self.index_df = (
            self.values_df
            .drop_duplicates()
            .sort_values(list(self.columns))
            .reset_index(drop = True)
        )

        self.levels = [row for row in self.index_df.itertuples(index = False)]
        self.indices = {}
        for i, level in enumerate(self.levels):
            self.indices[level] = i

    def get_level(self, i):
        return self.levels[i]

    def get_index(self, level):
        return self.indices[level]

    def get_stan_name(self):
        return '_'.join(self.columns)

    def code(self):
        size = len(self.source_df.index)
        return f"int {self.get_stan_name()}[{size}];"

class StanData:
    def __init__(self, series):
        self.series = series

    def get_stan_name(self):
        return self.series.name

    def code(self):
        dtype = self.series.dtype
        size = len(self.series.index)

        if pandas.api.types.is_integer_dtype(dtype):
            return f"int {self.get_stan_name()}[{size}];"
        elif pandas.api.types.is_float_dtype(dtype):
            return f"vector[{size}] {self.get_stan_name()};"


class StanParam:
    def __init__(self, name, stan_index = None):
        self.name = name
        self.stan_index = stan_index

    def get_stan_name(self):
        return self.name

    def code(self):
        if self.stan_index is not None:
            size = len(self.stan_index.levels)

            return f"vector[{size}] {self.get_stan_name()};"
        else:
            return f"real {self.get_stan_name()};"
