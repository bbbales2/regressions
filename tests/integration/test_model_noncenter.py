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
    y' ~ normal(theta[school], sigma);
    theta' = mu + z[school] * tau;
    z[school] ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    """

    return Model(data_df, model_string=model_string)


@pytest.fixture
def optimization_fit(eight_schools_model):
    return eight_schools_model.optimize(init=0.1)


@pytest.fixture
def sample_fit(eight_schools_model):
    return eight_schools_model.sample(init=0.1, num_draws=1000, num_warmup=1000)


def test_optimize_eight_schools(optimization_fit):
    mu_df = optimization_fit.draws("mu")
    z_df = optimization_fit.draws("z")
    tau_df = optimization_fit.draws("tau")

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

    assert mu_df["mu"][0] == pytest.approx(4.61934000, rel=1e-2)
    assert tau_df["tau"][0] == pytest.approx(0.36975800, rel=1e-2)
    assert z_df["z"].to_list() == pytest.approx(ref_z, rel=1e-2, abs=1e-3)


def test_sample_eight_schools(sample_fit):
    mu_diag_df = sample_fit.diag("mu")
    tau_diag_df = sample_fit.diag("tau")
    z_diag_df = sample_fit.diag("z")
    theta_diag_df = sample_fit.diag("theta")

    assert (mu_diag_df["ess"] > 1000).all()
    assert (tau_diag_df["ess"] > 1000).all()
    assert (z_diag_df["ess"] > 1000).all()
    assert (theta_diag_df["ess"] > 1000).all()

    assert (mu_diag_df["rhat"] < 1.01).all()
    assert (tau_diag_df["rhat"] < 1.01).all()
    assert (z_diag_df["rhat"] < 1.01).all()
    assert (theta_diag_df["rhat"] < 1.01).all()

    mu_df = sample_fit.draws("mu")
    tau_df = sample_fit.draws("tau")
    z_df = sample_fit.draws("z")
    theta_df = sample_fit.draws("theta")
    z_mean_df = z_df.groupby("school").agg({"z": numpy.mean}).reset_index().sort_values("school")
    theta_mean_df = theta_df.groupby("school").agg({"theta": numpy.mean}).reset_index().sort_values("school")

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

    assert mu_df["mu"].mean() == pytest.approx(4.60, rel=1e-1)
    assert tau_df["tau"].mean() == pytest.approx(1.37, rel=1e-1)
    assert z_mean_df["z"].to_list() == pytest.approx(ref_z_mean, rel=1e-1, abs=1e-1)
    assert theta_mean_df["theta"].to_list() == pytest.approx(ref_theta_mean, rel=1e-1, abs=1e-1)


def test_optimization_draws(optimization_fit):
    mu_df = optimization_fit.draws("mu")
    tau_df = optimization_fit.draws("tau")
    z_df = optimization_fit.draws("z")
    theta_df = optimization_fit.draws("theta")

    mu_tau_df = optimization_fit.draws(("mu", "tau"))
    mu_theta_df = optimization_fit.draws(("mu", "theta"))
    z_theta_df = optimization_fit.draws(("z", "theta"))

    pandas.testing.assert_frame_equal(mu_tau_df, pandas.merge(mu_df, tau_df, how="outer"))
    pandas.testing.assert_frame_equal(mu_theta_df, pandas.merge(mu_df, theta_df, how="outer"))
    pandas.testing.assert_frame_equal(z_theta_df, pandas.merge(z_df, theta_df, how="outer"))


def test_sample_draws(sample_fit):
    mu_df = sample_fit.draws("mu")
    tau_df = sample_fit.draws("tau")
    z_df = sample_fit.draws("z")
    theta_df = sample_fit.draws("theta")

    mu_tau_df = sample_fit.draws(("mu", "tau"))
    mu_theta_df = sample_fit.draws(("mu", "theta"))
    z_theta_df = sample_fit.draws(("z", "theta"))

    pandas.testing.assert_frame_equal(mu_tau_df, pandas.merge(mu_df, tau_df, how="outer"))
    pandas.testing.assert_frame_equal(mu_theta_df, pandas.merge(mu_df, theta_df, how="outer"))
    pandas.testing.assert_frame_equal(z_theta_df, pandas.merge(z_df, theta_df, how="outer"))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
