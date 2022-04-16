import logging
import os
import pathlib
import pandas
import pytest
import time

from rat import ast
from rat.compiler import CompileError
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_time_series():
    data_df = pandas.read_csv(os.path.join(test_dir, "time_series.csv"))

    model_string = """
    y' ~ normal(mu[i], 0.1);
    mu[i]' ~ normal(mu[shift(i, 1)], 0.3);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=0.1)
    mu_df = fit.draws("mu")

    mu_ref = [
        -0.4372130,
        -0.2322210,
        -0.1530660,
        -0.6052980,
        -0.8591810,
        -0.6521860,
        -0.4640860,
        -0.0444671,
        -0.1890310,
        0.0243282,
    ]

    assert mu_df["mu"].to_list() == pytest.approx(mu_ref, rel=1e-2)


def test_optimize_time_series_non_center():
    data_df = pandas.read_csv(os.path.join(test_dir, "time_series.csv"))

    model_string = """
    y' ~ normal(mu[i], 0.1);
    mu[i]' = mu[shift(i, 1)] + epsilon[i];
    epsilon[i] ~ normal(0, 0.3);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=0.1)
    mu_df = fit.draws("mu")

    mu_ref = [
        -0.4372130,
        -0.2322210,
        -0.1530660,
        -0.6052980,
        -0.8591810,
        -0.6521860,
        -0.4640860,
        -0.0444671,
        -0.1890310,
        0.0243282,
    ]

    assert mu_df["mu"].to_list() == pytest.approx(mu_ref, rel=1e-2)


def test_optimize_time_series_2():
    data_df = pandas.read_csv(os.path.join(test_dir, "time_series_2.csv"))

    model_string = """
    score_diff' ~ normal(skills[year, team1] - skills[year, team2], sigma);
    skills[year, team]' ~ normal(skills[shift(year, 1), team], 0.5);
    sigma<lower = 0.0> ~ normal(0, 1.0);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=0.1, tolerance=1e-1)
    skills_df = fit.draws("skills")
    sigma_df = fit.draws("sigma")

    team = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    year = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    skills_ref = [
        0.71583100,
        0.04974240,
        -0.77518200,
        0.70642400,
        -0.18468000,
        -0.52541500,
        -0.02595790,
        -0.29738500,
        0.32596500,
        -0.80349400,
        0.00426407,
        0.80181100,
        -0.73733100,
        0.13827900,
        0.58915900,
    ]
    skills_ref_df = pandas.DataFrame(zip(team, year, skills_ref), columns=["team", "year", "skills_ref"])
    joined_df = skills_df.merge(skills_ref_df, on=["team", "year"], how="left", validate="one_to_one")

    assert sigma_df["sigma"][0] == pytest.approx(3.51620000, rel=1e-2)
    assert joined_df["skills"].to_list() == pytest.approx(joined_df["skills_ref"].to_list(), abs=1e-1)


def test_optimize_time_series_2_non_center():
    data_df = pandas.read_csv(os.path.join(test_dir, "time_series_2.csv"))

    model_string = """
    score_diff' ~ normal(skills[team1, year] - skills[team2, year], sigma);
    skills[team, year]' = skills[team, shift(year, 1)] + epsilon[team, year] * tau;
    epsilon[team, year] ~ normal(0.0, 1.0);
    tau = 0.5;
    sigma<lower = 0.0> ~ normal(0, 1.0);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=0.1, chains=1)
    skills_df = fit.draws("skills")
    sigma_df = fit.draws("sigma")

    team = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    year = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    skills_ref = [
        0.71583100,
        0.04974240,
        -0.77518200,
        0.70642400,
        -0.18468000,
        -0.52541500,
        -0.02595790,
        -0.29738500,
        0.32596500,
        -0.80349400,
        0.00426407,
        0.80181100,
        -0.73733100,
        0.13827900,
        0.58915900,
    ]
    skills_ref_df = pandas.DataFrame(zip(team, year, skills_ref), columns=["team", "year", "skills_ref"])
    joined_df = skills_df.merge(skills_ref_df, on=["team", "year"], how="left", validate="one_to_one")

    assert sigma_df["sigma"][0] == pytest.approx(3.51620000, rel=1e-2)
    assert joined_df["skills"].to_list() == pytest.approx(joined_df["skills_ref"].to_list(), abs=1e-1)

def test_optimize_time_series_2_non_center_infer_tau():
    data_df = pandas.read_csv(os.path.join(test_dir, "time_series_2.csv"))

    model_string = """
    score_diff' ~ normal(skills[team1, year] - skills[team2, year], sigma);
    skills[team, year]' = skills[team, shift(year, 1)] + epsilon[team, year] * tau;
    epsilon[team, year] ~ normal(0.0, 1.0);
    tau<lower = 0.0> ~ normal(0.0, 0.5);
    sigma<lower = 0.0> ~ normal(0, 1.0);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=0.1, chains=1)

def test_optimize_time_series_2_error():
    data_df = pandas.read_csv(os.path.join(test_dir, "time_series_2.csv"))

    model_string = """
    score_diff' ~ normal(skills[team1, year] - skills[team2, year], sigma);
    skills' ~ normal(skills[team, shift(year, 1)], 0.5);
    sigma<lower = 0.0> ~ normal(0, 1.0);
    """

    with pytest.raises(CompileError, match="not found"):
        model = Model(data_df, model_string=model_string)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
