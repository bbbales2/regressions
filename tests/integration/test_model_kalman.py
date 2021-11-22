import os
import pathlib
import pandas
import pytest
import time

from rat import ops
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_kalman():
    data_df = pandas.read_csv(os.path.join(test_dir, "kalman.csv"))

    model_string = """
    y ~ normal(mu[i], 0.1);
    mu[i] ~ normal(mu[shift(i, 1)], 0.3);
    """

    # parsed_lines = [
    #    ops.Normal(
    #        ops.Data("y"), ops.Param("mu", ops.Index(("i",))), ops.RealConstant(0.1)
    #    ),
    #    ops.Normal(
    #        ops.Param("mu", ops.Index(("i",))),
    #        ops.Param("mu", ops.Index(("i",), shifts=(1,))),
    #        ops.RealConstant(0.3),
    #    ),
    # ]

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

    assert mu_df["value"].to_list() == pytest.approx(mu_ref, rel=1e-2)


def test_optimize_kalman_2():
    data_df = pandas.read_csv(os.path.join(test_dir, "kalman_2.csv"))

    model_string = """
    score_diff ~ normal(skills[team1, year] - skills[team2, year], sigma);
    skills[team, year] ~ normal(skills[team, shift(year, 1)], 0.5);
    sigma<lower = 0.0> ~ normal(0, 1.0);
    """

    # parsed_lines = [
    #    ops.Normal(
    #        ops.Data("score_diff"),
    #        ops.Diff(
    #            ops.Param("skills", ops.Index(("team1", "year"))),
    #            ops.Param("skills", ops.Index(("team2", "year"))),
    #        ),
    #        ops.Param("sigma"),
    #    ),
    #    ops.Normal(
    #        ops.Param("skills", ops.Index(("team", "year"))),
    #        ops.Param("skills", ops.Index(("team", "year"), shifts=(None, 1))),
    #        ops.RealConstant(0.5),
    #    ),
    #    ops.Normal(
    #        ops.Param("sigma", lower=ops.RealConstant(0.0)),
    #        ops.RealConstant(0.0),
    #        ops.RealConstant(1.0),
    #    ),
    # ]

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=0.1)
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
    skills_ref_df = pandas.DataFrame(zip(team, year, skills_ref), columns=["team", "year", "value_ref"])
    joined_df = skills_df.merge(skills_ref_df, on=["team", "year"], how="left", validate="one_to_one")

    assert sigma_df["value"][0] == pytest.approx(3.51620000, rel=1e-2)
    assert joined_df["value"].to_list() == pytest.approx(joined_df["value_ref"].to_list(), abs=1e-1)


if __name__ == "__main__":
    pytest.main([__file__])