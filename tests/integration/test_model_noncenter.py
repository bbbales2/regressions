import logging
import numpy
import os
import pathlib
import pandas
import pytest

from rat import ops
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


@pytest.fixture
def eight_schools_model():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = """
    y ~ normal(theta[school], sigma[school]);
    theta[school] = mu + z[school] * tau;
    z[school] ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ lognormal(0, 1);
    """

    # TODO: Add a unit test that the thing above parses to the thing below
    parsed_lines = [
        ops.Normal(ops.Data("y"), ops.Param("theta", ops.Index(("school",))), ops.Data("sigma")),
        ops.Assignment(
            ops.Param("theta", ops.Index(("school",))),
            ops.Sum(ops.Param("mu"), ops.Mul(ops.Param("z", ops.Index(("school",))), ops.Param("tau"))),
        ),
        ops.Normal(ops.Param("mu"), ops.RealConstant(0.0), ops.RealConstant(5.0)),
        ops.Normal(ops.Param("z", ops.Index(("school",))), ops.RealConstant(0.0), ops.RealConstant(1.0)),
        ops.LogNormal(ops.Param("tau", lower=ops.RealConstant(0.0)), ops.RealConstant(0.0), ops.RealConstant(1.0)),
    ]

    return Model(data_df, parsed_lines)  # model_string=model_string


def test_optimize_eight_schools(eight_schools_model):
    fit = eight_schools_model.optimize(init=0.1)
    mu_df = fit.draws("mu")
    z_df = fit.draws("z")
    tau_df = fit.draws("tau")

    ref_z = [
        0.03842000,
        0.01246530,
        -0.01098500,
        0.00726350,
        -0.02562610,
        -0.01106250,
        0.04939970,
        0.00839282,
    ]
   
    ref_theta_mean = [
        4.63356000,
        4.62396000,
        4.61529000,
        4.62204000,
        4.60988000,
        4.61526000,
        4.63762000,
        4.62246000,
    ]

    assert mu_df["value"][0] == pytest.approx(4.61934000, rel=1e-2)
    assert tau_df["value"][0] == pytest.approx(0.36975800, rel=1e-2)
    assert z_df["value"].to_list() == pytest.approx(ref_z, rel=1e-2, abs=1e-3)


def test_sample_eight_schools(eight_schools_model):
    fit = eight_schools_model.sample(init=0.1, num_draws=1000, num_warmup=1000)

    mu_diag_df = fit.diag("mu")
    tau_diag_df = fit.diag("tau")
    z_diag_df = fit.diag("z")
    theta_diag_df = fit.diag("theta")

    assert (mu_diag_df["ess"] > 1000).all()
    assert (tau_diag_df["ess"] > 1000).all()
    assert (z_diag_df["ess"] > 1000).all()
    assert (theta_diag_df["ess"] > 1000).all()

    assert (mu_diag_df["rhat"] < 1.01).all()
    assert (tau_diag_df["rhat"] < 1.01).all()
    assert (z_diag_df["rhat"] < 1.01).all()
    assert (theta_diag_df["rhat"] < 1.01).all()

    mu_df = fit.draws("mu")
    tau_df = fit.draws("tau")
    z_df = fit.draws("z")
    theta_df = fit.draws("theta")
    z_mean_df = z_df.groupby("school").agg({"value": numpy.mean}).reset_index().sort_values("school")
    theta_mean_df = theta_df.groupby("school").agg({"value": numpy.mean}).reset_index().sort_values("school")

    ref_z_mean = [
        0.137,
        0.033,
        -0.027,
        0.023,
        -0.077,
        -0.032,
        0.185,
        0.021,
    ]

    ref_theta_mean = [
        4.97,
        4.67,
        4.53,
        4.70,
        4.40,
        4.50,
        5.05,
        4.76,
    ]

    assert mu_df["value"].mean() == pytest.approx(4.60, rel=1e-1)
    assert tau_df["value"].mean() == pytest.approx(1.37, rel=1e-1)
    assert z_mean_df["value"].to_list() == pytest.approx(ref_z_mean, rel=1e-1, abs=1e-1)
    assert theta_mean_df["value"].to_list() == pytest.approx(ref_theta_mean, rel=1e-1, abs=1e-1)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
