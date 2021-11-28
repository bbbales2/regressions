import logging
import numpy
import os
import pathlib
import pandas
import pytest

from rat import ops
from rat import compiler
from rat.model import Model
from rat.parser import Parser
from rat.scanner import scanner
from pathlib import Path

test_dir = pathlib.Path(__file__).parent


def test_sample_normal_mu():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    model_string = """
    y ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.sample(num_draws=200)
    mu_df = fit.draws("mu")

    import plotnine

    (plotnine.ggplot(mu_df) + plotnine.geom_point(plotnine.aes("draw", "value", color="chain"))).draw(show=True)

    print(mu_df.agg({"value": [numpy.mean, numpy.std]}))
    print(fit.ess("mu"))
    print(fit.rhat("mu"))
    # assert mu_df["value"][0] == pytest.approx(-1.11249, rel=1e-2)


def adsftest_full():
    data_df = (
        pandas.read_csv(os.path.join(test_dir, "games_small.csv"))
        .assign(score_diff=lambda df: (df.home_score - df.away_score).astype("float"))
        .assign(year=lambda df: df["date"].str[0:4].astype("int"))
    )

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
    # skills_mu[year] ~ normal(0.0, 1.0);
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
    fit = model.sample(num_draws=20)

    tau_df = fit.draws("tau")
    skills_df = fit.draws("skills")

    # print(tau_df)
    # print(skills_df)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
