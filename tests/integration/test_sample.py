import logging
import numpy
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model
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

    model_string = """
    score_diff' ~ normal(skills[home_team, year] - skills[away_team, year], sigma);
    skills[team, year]' ~ normal(skills_mu[year], tau);
    skills_mu[year] ~ normal(0.0, 1.0);
    tau<lower = 0.0> ~ normal(0.0, 1.0);
    sigma<lower = 0.0> ~ normal(0.0, 1.0);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.sample(num_draws=20, num_warmup=201)

    tau_df = fit.draws("tau")
    skills_df = fit.draws("skills")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
