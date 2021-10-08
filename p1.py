import pandas
import typing

import context
import ops

df = (
    pandas.read_csv("games_small.csv")
    .assign(score_diff = lambda df : (df.home_score - df.away_score).astype("float"))
    .assign(year = lambda df : df["date"].str[0:4].astype("int"))
)

trees = [
    ops.Normal(
        ops.Data("score_diff"),
        ops.Diff(
            ops.Param("skills", ops.Index("home_team", "year")),
            ops.Param("skills", ops.Index("away_team", "year"))
        ),
        ops.Param("sigma")
    ),
    ops.Normal(
        ops.Param("skills", ops.Index("team", "year")),
        ops.Param("skills_mu", ops.Index("year")),
        ops.Param("tau")
    ),
    ops.Normal(
        ops.Param("skills_mu", ops.Index("year_mu")),
        ops.RealConstant(0.0),
        ops.RealConstant(1.0)
    ),
    ops.Normal(
        ops.Param("tau"),
        ops.RealConstant(0.0),
        ops.RealConstant(1.0)
    )
]

#score_diff ~ normal(skills[home_team, year] - skills[away_team, year], sigma);
#skills[team, year] ~ normal(skills_mu[year], tau);
#tau ~ normal(0.0, 1.0);
#sigma ~ normal(0.0, 10.0);

stan_data : typing.Dict[str, context.StanData] = {}
stan_params : typing.Dict[str, context.StanParam] = {}

data_lines : typing.List[str] = []
parameters_lines : typing.List[str] = []
model_lines : typing.List[str] = []

for tree in trees:
    assert isinstance(tree, ops.Distr)

    if isinstance(tree.lhs, ops.Data):
        tree_df = df
    else:
        param = stan_params[tree.lhs.get_key()]
        param_index = param.stan_index
        if param_index is not None:
            tree_df = param_index.index_df
            if tree.lhs.index is not None:
                tree_df.columns = tree.lhs.index.get_key()
        else:
            tree_df = None

    stan_indices : typing.Dict[tuple, context.StanIndex] = {}

    #for index in ops.search_tree(ops.Index, tree):
    #    index_key = index.get_key()
    #    if index_key not in stan_indices:
    #        stan_indices[index_key] = (
    #            context.StanIndex(tree_df[list(index_key)])
    #        )

    for datum in ops.search_tree(ops.Data, tree):
        datum_key = datum.get_key()
        if datum_key not in stan_data:
            stan_data[datum_key] = context.StanData(tree_df[datum.name])

    param_indices = {}
    for param in ops.search_tree(ops.Param, tree):
        param_key = param.get_key()
        if param.index is not None:
            if param_key not in param_indices:
                param_indices[param_key] = []
            param_indices[param_key].append(param.index.get_key())
        else:
            stan_params[param_key] = context.StanParam(param_key)

    for param_key, index_key_set in param_indices.items():
        columns = list(index_key_set[0])
        value_dfs = []
        for key in index_key_set:
            value_df = tree_df.loc[:, key]
            value_df.columns = columns
            value_dfs.append(value_df)

        values_df = pandas.concat(value_dfs, ignore_index = True)

        for key in index_key_set:
            index_df = tree_df.loc[:, key]
            stan_index = context.StanIndex(index_df, values_df)
            stan_indices[key] = stan_index

        stan_params[param_key] = context.StanParam(param_key, stan_index)

    for expr in ops.search_tree(ops.Index, tree):
        expr.populate(stan_indices[expr.get_key()])

    for expr in ops.search_tree(ops.Data, tree):
        expr.populate(stan_data[expr.get_key()])

    for expr in ops.search_tree(ops.Param, tree):
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

#score_diff ~ normal(skills[home_team, year] - skills[away_team, year], sigma);
#skills[team, year] ~ normal(skills_mu[year], tau);
#tau ~ normal(0.0, 1.0);
#sigma ~ normal(0.0, 10.0);




#for index_tuple in indices:
#    for row in df[list(index_tuple)].itertuples(index = False):
#        print(index_tuple, row, multifactors[index_tuple].get_index(row))

#for column, series in df.iteritems():
#    is_int = pandas.api.types.is_integer_dtype(series)
#    is_float = pandas.api.types.is_integer_dtype(series)
#    is_str = pandas.api.types.is_string_dtype(series)
#    if is_int or is_str:
#        factors[name] = make_factor(series.tolist())
        
