from typing import List, Dict
import tokens
import variables
import pandas


def parse(data_df: pandas.DataFrame, tokenized_lines: List[tokens.Expr]):
    data_uses: Dict[str, variables.DataUse] = {}
    parameter_uses: Dict[str, variables.ParamUse] = {}
    data_variables: Dict[str, variables.Data] = {}
    parameter_variables: Dict[str, variables.Param] = {}

    data_lines: List[str] = []
    parameters_lines: List[str] = []
    model_lines: List[str] = []

    for line in tokenized_lines:
        assert isinstance(line, tokens.Distr)
        index_uses: Dict[tuple, variables.IndexUse] = {}
        index_variables: Dict[tuple, variables.Index] = {}

        if isinstance(line.lhs, tokens.Data):
            # If the left hand side is data, the dataframe comes from input
            line_df = data_df
        else:
            # Otherwise, the dataframe comes from the parameter (unless it's scalar then it's none)
            parameter = parameter_variables[line.lhs.get_key()]
            index = parameter.index
            if index is not None:
                line_df = index.df.copy()
                # Rename columns to match names given on lhs
                if line.lhs.index is not None:
                    line_df.columns = line.lhs.index.get_key()
            else:
                line_df = None

        for data in tokens.search_tree(tokens.Data, line):
            data_key = data.get_key()
            if data_key not in data_variables:
                data_variables[data_key] = variables.Data(data_key, line_df[data_key])

        parameter_index_keys: Dict[str, List[variables.Index]] = {}
        # Find all the ways that each parameter is indexed
        for parameter in tokens.search_tree(tokens.Param, line):
            parameter_key = parameter.get_key()
            if parameter.index is not None:
                if parameter_key not in parameter_index_keys:
                    parameter_index_keys[parameter_key] = []
                parameter_index_keys[parameter_key].append(parameter.index.get_key())

        # Build the parameters index variables
        for parameter_key, index_key_list in parameter_index_keys.items():
            columns = list(index_key_list[0])
            value_dfs = []
            for index_key in index_key_list:
                value_df = line_df.loc[:, index_key]
                value_df.columns = columns # columns must be the same to concat
                value_dfs.append(value_df)

            values_df = pandas.concat(value_dfs, ignore_index = True)
            index = variables.Index(values_df)
            index_variables[parameter_key] = index
            parameter_variables[parameter_key] = variables.Param(parameter_key, index)

        # Build the parameters
        for parameter in tokens.search_tree(tokens.Param, line):
            parameter_key = parameter.get_key()
            if parameter.index is not None:
                index = index_variables[parameter_key]
                index_df = line_df.loc[:, parameter.index.get_key()]
                parameter_usesvariables[parameter_key] = variables.Param(parameter_key, index_df, index)
            else:
                parameter_variables[parameter_key] = variables.Param(parameter_key)

            for index_key in index_key_list:
                index_df = line_df.loc[:, index_key]
                index_use = variables.IndexUse(index_key, index_df, index)
                index_use_variables[index_key] = index_use
                parameter_variables[parameter_key] = variables.Param(parameter_key, index_use)



        for expr in tokens.search_tree(tokens.Index, tree):
            expr.populate(stan_indices[expr.get_key()])

        for expr in tokens.search_tree(tokens.Data, tree):
            expr.populate(stan_data[expr.get_key()])

        for expr in tokens.search_tree(tokens.Param, tree):
            expr.populate(stan_params[expr.get_key()])

        for stan_index in stan_indices.values():
            data_lines.append(stan_index.code())

        model_lines.append(tree.code())

for stan_datum in stan_data.values():
    data_lines.append(stan_datum.code())
for stan_param in stan_params.values():
    parameters_lines.append(stan_param.code())

data_block = "\n".join(data_lines)
parameters_block = "\n".join(parameters_lines)
model_block = "\n".join(model_lines)

print(f"""
data {{
{data_block}
}}
parameters {{
{parameters_block}
}}
model {{
{model_block}
}}
""")

# score_diff ~ normal(skills[home_team_away_team_idx] - skills[away_team_idx], sigma)

# for index_tuple in indices:
#    for row in df[list(index_tuple)].itertuples(index = False):
#        print(index_tuple, row, multifactors[index_tuple].get_index(row))

# for column, series in df.iteritems():
#    is_int = pandas.api.types.is_integer_dtype(series)
#    is_float = pandas.api.types.is_integer_dtype(series)
#    is_str = pandas.api.types.is_string_dtype(series)
#    if is_int or is_str:
#        factors[name] = make_factor(series.tolist())
