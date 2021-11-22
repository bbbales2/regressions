import os
import pathlib
import pandas

from rat import ops
from rat import compiler
from rat.model import Model
from rat.parser import Parser
from rat.scanner import scanner
from pathlib import Path

test_dir = pathlib.Path(__file__).parent


def test_full():
    data_df = (
        pandas.read_csv(os.path.join(test_dir, "games_small.csv"))
        .assign(score_diff=lambda df: (df.home_score - df.away_score).astype("float"))
        .assign(year=lambda df: df["date"].str[0:4].astype("int"))
    )

    print(data_df)
    print(data_df.columns)

    parsed_lines = [
        ops.Normal(
            ops.Data("score_diff"),
            ops.Diff(
                ops.Param("skills", ops.Index(("home_team", "year"))),
                ops.Param("skills", ops.Index(("away_team", "year"))),
            ),
            ops.Param("sigma"),
        ),
        ops.Normal(
            ops.Param("skills", ops.Index(("team", "year"))),
            ops.Param("skills_mu", ops.Index(("year",))),
            ops.Param("tau"),
        ),
        ops.Normal(
            ops.Param(
                "skills_mu",
                ops.Index(("year",)),
            ),
            ops.RealConstant(0.0),
            ops.RealConstant(1.0),
        ),
        ops.Normal(
            ops.Param("tau", lower=ops.RealConstant(0.0)),
            ops.RealConstant(0.0),
            ops.RealConstant(1.0),
        ),
        ops.Normal(
            ops.Param("sigma", lower=ops.RealConstant(0.0)),
            ops.RealConstant(0.0),
            ops.RealConstant(1.0),
        ),
    ]

    # input_str = """
    # score_diff~normal(skills[home_team, year]-skills[away_team, year],sigma);
    # skills[team, year] ~ normal(skills_mu[lag(year)], tau);
    # tau<lower=0.0> ~ normal(0.0, 1.0);
    # sigma<lower=0.0> ~ normal(0.0, 10.0);
    # """

    # parsed_lines = []
    # for line in input_str.split("\n"):
    #     if not line: continue
    #     parsed_lines.append(Parser(scanner(line), data_df.columns).statement())

    # import pprint
    # pprint.pprint(parsed_lines)

    model = Model(data_df, parsed_lines)
    fit = model.sample(20)

    tau_df = fit.draws("tau")
    skills_df = fit.draws("skills")

    # print(tau_df)
    # print(skills_df)
