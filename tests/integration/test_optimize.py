import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_normal_mu():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    model_string = """
    y ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    ## TODO: Add a unit test that the thing above parses to the thing below
    # parsed_lines = [
    #    ops.Normal(ops.Data("y"), ops.Param("mu"), ops.RealConstant(1.5)),
    #    ops.Normal(ops.Param("mu"), ops.RealConstant(-0.5), ops.RealConstant(0.3)),
    # ]

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=2.0)
    mu_df = fit.draws("mu")

    assert mu_df["mu"][0] == pytest.approx(-1.11249, rel=1e-2)


def test_optimize_normal_mu_sigma():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    model_string = """
    y ~ normal(mu, sigma);
    mu ~ normal(-0.5, 0.3);
    sigma<lower = 0.0> ~ normal(0.0, 0.7);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=2.0)
    mu_df = fit.draws("mu")
    sigma_df = fit.draws("sigma")

    assert mu_df["mu"][0] == pytest.approx(-0.837757, rel=1e-2)
    assert sigma_df["sigma"][0] == pytest.approx(3.880730, rel=1e-2)


def test_optimize_normal_mu_sigma_bounded():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    model_string = """
    y ~ normal(mu, sigma);
    mu<lower = 0.0> ~ normal(-0.5, 0.3);
    sigma<lower = 0.0, upper = 0.1> ~ normal(0.0, 0.7);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=2.0)
    mu_df = fit.draws("mu")
    sigma_df = fit.draws("sigma")

    # With different constraints, we would get solutions
    # outside these bounds
    assert mu_df["mu"][0] >= 0.0
    assert sigma_df["sigma"][0] <= 0.1 + 1e-7  # not sure how solution is slightly above, guess some precision thing?


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
