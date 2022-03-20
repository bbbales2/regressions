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
from pathlib import Path

test_dir = pathlib.Path(__file__).parent


def test_sample_normal_mu():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    model_string = """
    y ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.sample(num_draws=1000)
    mu_df = fit.draws("mu")
    leapfrog_steps_df = fit.draws("__leapfrog_steps")
    divergences_df = fit.draws("__divergences")

    assert (leapfrog_steps_df["__leapfrog_steps"] > 0).all()
    assert (divergences_df["__divergences"] == 0).all()
    assert mu_df["mu"].mean() == pytest.approx(-1.11, rel=1e-2)


def test_sample_normal_mu_thin():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    model_string = """
    y ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    num_draws = 250
    chains = 4
    thin = 4

    model = Model(data_df, model_string=model_string)
    fit = model.sample(num_draws=num_draws, chains=chains, thin=thin)
    mu_df = fit.draws("mu")

    assert len(mu_df) == num_draws * chains
    assert mu_df["mu"].mean() == pytest.approx(-1.11, rel=1e-2)


def test_sample_acceptance_rate():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    model_string = """
    y ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.sample(num_draws=1000, target_acceptance_rate=0.99)
    mu_df = fit.draws("mu")

    assert mu_df["mu"].mean() == pytest.approx(-1.11, rel=1e-2)


def test_full():
    data_df = (
        pandas.read_csv(os.path.join(test_dir, "games_small.csv"))
        .assign(score_diff=lambda df: (df.home_score - df.away_score).astype("float"))
        .assign(year=lambda df: df["date"].str[0:4].astype("int"))
    )

    parsed_lines = [
        ops.Normal(
            ops.Data("score_diff", prime=True),
            ops.Diff(
                ops.Param("skills", subscript=ops.Subscript(("home_team", "year"))),
                ops.Param("skills", subscript=ops.Subscript(("away_team", "year"))),
            ),
            ops.Param("sigma"),
        ),
        ops.Normal(
            ops.Param("skills", prime=True, subscript=ops.Subscript(("team", "year"))),
            ops.Param("skills_mu", subscript=ops.Subscript(("year",))),
            ops.Param("tau"),
        ),
        ops.Normal(
            ops.Param(
                "skills_mu",
                subscript=ops.Subscript(("year",)),
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

    model = Model(data_df, parsed_lines=parsed_lines)
    fit = model.sample(num_draws=20, num_warmup=201)

    tau_df = fit.draws("tau")
    skills_df = fit.draws("skills")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
