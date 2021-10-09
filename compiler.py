from typing import List, Dict, Set
import ops
import variables
import pandas

class LineFunction:
    data_variables: List[variables.Data]
    index_use_variables: List[variables.IndexUse]
    parameter_variables: List[variables.Param]
    body_code : str

    def args(self):
        return [variable.code() for variable in self.data_variables + self.index_use_variables + self.parameter_variables]

    def code(self):
        return "\n".join(
            f"def func({','.join(self.args())}):",
            f"  {self.body_code}"
        )
    
    def call_code(self):
        return f"func({','.join(self.args())})"

def compile(data_df: pandas.DataFrame, parsed_lines: List[ops.Expr]):
    data_variables: Dict[str, variables.Data] = {}
    parameter_variables: Dict[str, variables.Param] = {}
    index_variables: Dict[tuple, variables.Index] = {}

    line_functions = []

    for line in parsed_lines:
        assert isinstance(line, ops.Distr)
        data_variables_used: Set[str] = set()
        parameter_variables_used: Set[str] = set()
        index_use_variables: Dict[str, variables.IndexUse] = {}

        if isinstance(line.variate, ops.Data):
            # If the left hand side is data, the dataframe comes from input
            line_df = data_df
        else:
            # Otherwise, the dataframe comes from the parameter (unless it's scalar then it's none)
            parameter = parameter_variables[line.variate.get_key()]
            index = parameter.index
            if index is not None:
                line_df = index.df.copy()
                # Rename columns to match names given on the lhs
                if line.variate.index is not None:
                    line_df.columns = line.variate.index.get_key()
            else:
                line_df = None

        for data in ops.search_tree(ops.Data, line):
            data_key = data.get_key()
            data_variables_used.add(data_key)
            if data_key not in data_variables:
                data_variables[data_key] = variables.Data(data_key, line_df[data_key])

        parameter_index_keys: Dict[str, List[variables.Index]] = {}
        # Find all the ways that each parameter is indexed
        for parameter in ops.search_tree(ops.Param, line):
            parameter_key = parameter.get_key()
            parameter_variables_used.add(parameter_key)
            if parameter_key not in parameter_index_keys:
                parameter_index_keys[parameter_key] = []

            if parameter.index is None:
                parameter_index_keys[parameter_key].append(None)
            else:
                parameter_index_keys[parameter_key].append(parameter.index.get_key())

        # Build the parameters
        for parameter_key, index_key_list in parameter_index_keys.items():
            any_none = any(key is None for key in index_key_list)
            all_none = all(key is None for key in index_key_list)
            if any_none:
                # scalar parameters have None has the index
                if all_none:
                    parameter_variables[parameter_key] = variables.Param(parameter_key)
                else:
                    raise Exception("Scalar parameters don't support indexing")
            else:
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

        for parameter in ops.search_tree(ops.Param, line):
            parameter_key = parameter.get_key()
            if parameter.index is not None:
                index_key = parameter.index.get_key()
                index = index_variables[parameter_key]
                index_df = line_df.loc[:, parameter.index.get_key()]
                index_use_variables[index_key] = variables.IndexUse(index_key, index_df, index)
                #parameter_uses[parameter_key] = variables.ParamUse(param, index_df, index)

        line_function = LineFunction(
            [data_variables[name] for name in data_variables_used],
            index_use_variables.values(),
            [parameter_variables[name] for name in parameter_variables_used],
            line.code()
        )

        line_functions.append(line_function)
    
    for key, data in data_variables.items():
        data.initialize(locals())
    
    for key, parameter in parameter_variables.items():
        parameter.initialize(locals())

    # IN WRONG SCOPE
    for key, index in index_variables.items():
        index.initialize(locals())
# score_diff = df["score_diff"]
# sigma = 0.0
# home_team_year = computed_index
# away_team_year = computed_index
# skills = jnp.zeros(N)
# def line1_lpdf(score_diff, sigma, home_team_year, away_team_year, skills)
#     return jnp.sum(normal_lpdf(score_diff, skills[home_team_year] - skills[away_team_year], sigma))
# line1_lpdf_vec = vmap(line1_lpdf, (0, None, 0, 0, None), 0)
# ops.Normal(
#         ops.Data("score_diff"),
#         ops.Diff(
#             ops.Param("skills", ops.Index(("home_team", "year"))),
#             ops.Param("skills", ops.Index(("away_team", "year")))
#         ),
#         ops.Param("sigma")
#     )

#         for expr in ops.search_tree(ops.Index, tree):
#             expr.populate(stan_indices[expr.get_key()])

#         for expr in ops.search_tree(ops.Data, tree):
#             expr.populate(stan_data[expr.get_key()])

#         for expr in ops.search_tree(ops.Param, tree):
#             expr.populate(stan_params[expr.get_key()])

#         for stan_index in stan_indices.values():
#             data_lines.append(stan_index.code())

#         model_lines.append(tree.code())

# for stan_datum in stan_data.values():
#     data_lines.append(stan_datum.code())
# for stan_param in stan_params.values():
#     parameters_lines.append(stan_param.code())

# data_block = "\n".join(data_lines)
# parameters_block = "\n".join(parameters_lines)
# model_block = "\n".join(model_lines)

# print(f"""
# data {{
# {data_block}
# }}
# parameters {{
# {parameters_block}
# }}
# model {{
# {model_block}
# }}
# """)

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
