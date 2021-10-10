import numpy
import pandas
import jax
import jax.numpy as jnp
import blackjax
import blackjax.nuts

import ops
import compiler
from model import Model

data_df = (
    pandas.read_csv("games_small.csv")
    .assign(score_diff=lambda df: (df.home_score - df.away_score).astype("float"))
    .assign(year=lambda df: df["date"].str[0:4].astype("int"))
)

parsed_lines = [
    ops.Normal(
        ops.Data("score_diff"),
        ops.Diff(
            ops.Param("skills", ops.Index(("home_team", "year"))),
            ops.Param("skills", ops.Index(("away_team", "year")))
        ),
        ops.Param("sigma")
    ),
    ops.Normal(
        ops.Param("skills", ops.Index(("team", "year"))),
        ops.Param("skills_mu", ops.Index(("year",))),
        ops.Param("tau")
    ),
    ops.Normal(
        ops.Param("skills_mu", ops.Index(("year_mu",))),
        ops.Constant(0.0),
        ops.Constant(1.0)
    ),
    ops.Normal(
        ops.Param("tau"),
        ops.Constant(0.0),
        ops.Constant(1.0)
    )
]

model = Model(data_df, parsed_lines)

fit = model.sample(20)

tau_df = fit.draws("tau")
skills_df = fit.draws("skills")

print(tau_df)
print(skills_df)