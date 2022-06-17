import logging
import numpy
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_multiple_dataframes_eight_schools_optimize():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    y_data_df = data_df[["y", "school"]]
    sigma_data_df = data_df[["school", "sigma"]]

    model_string = """
    y' ~ normal(theta[school], sigma[school]);
    theta' = mu + z[school] * tau;
    z ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    """

    eight_schools_model = Model({"y_data": y_data_df, "sigma_data": sigma_data_df}, model_string=model_string)

    optimization_fit = eight_schools_model.optimize(init=0.1)

    mu_df = optimization_fit.draws("mu")
    theta_df = optimization_fit.draws("theta")
    tau_df = optimization_fit.draws("tau")

    ref_theta = [
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
    assert theta_df["theta"].to_list() == pytest.approx(ref_theta, rel=1e-2)


def test_multiple_dataframes_eight_schools_optimize_2():
    data1_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv")).rename(columns={"y": "y1", "sigma": "sigma1"})

    data2_df = data1_df.copy().rename(columns={"y1": "y2", "sigma1": "sigma2"})

    model_string = """
    y1' ~ normal(theta1[school], sigma1);
    theta1' = mu1 + z1[school] * tau1;
    z1 ~ normal(0, 1);
    mu1 ~ normal(0, 5);
    tau1<lower = 0.0> ~ log_normal(0, 1);

    y2' ~ normal(theta2[school], sigma2);
    theta2' = mu2 + z2[school] * tau2;
    z2 ~ normal(0, 1);
    mu2 ~ normal(0, 5);
    tau2<lower = 0.0> ~ log_normal(0, 1);
    """

    eight_schools_model = Model({"data1": data1_df, "data2": data2_df}, model_string=model_string)

    optimization_fit = eight_schools_model.optimize(init=0.1)

    ref_theta = [
        4.63356000,
        4.62396000,
        4.61529000,
        4.62204000,
        4.60988000,
        4.61526000,
        4.63762000,
        4.62246000,
    ]

    mu1_df = optimization_fit.draws("mu1")
    theta1_df = optimization_fit.draws("theta1")
    tau1_df = optimization_fit.draws("tau1")

    assert mu1_df["mu1"][0] == pytest.approx(4.61934000, rel=1e-2)
    assert tau1_df["tau1"][0] == pytest.approx(0.36975800, rel=1e-2)
    assert theta1_df["theta1"].to_list() == pytest.approx(ref_theta, rel=1e-2)

    mu2_df = optimization_fit.draws("mu2")
    theta2_df = optimization_fit.draws("theta2")
    tau2_df = optimization_fit.draws("tau2")

    assert mu2_df["mu2"][0] == pytest.approx(4.61934000, rel=1e-2)
    assert tau2_df["tau2"][0] == pytest.approx(0.36975800, rel=1e-2)
    assert theta2_df["theta2"].to_list() == pytest.approx(ref_theta, rel=1e-2)


def test_multiple_dataframes_eight_schools_error_to_many_to_few_rows():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    y_data_df = data_df[["y", "school"]]
    sigma_data_df = pandas.concat([data_df[["school", "sigma"]], data_df[["school", "sigma"]]])

    model_string = """
    y' ~ normal(theta[school], sigma[school]);
    theta' = mu + z[school] * tau;
    z ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    """

    with pytest.raises(Exception, match="unique"):
        Model({"y_data": y_data_df, "sigma_data": sigma_data_df}, model_string=model_string)

    sigma_data_df = data_df[["school", "sigma"]].iloc[
        :1,
    ]

    with pytest.raises(Exception, match="not defined"):
        Model({"y_data": y_data_df, "sigma_data": sigma_data_df}, model_string=model_string)


def test_multiple_dataframes_eight_schools_subscript_errors():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    y_data_df = data_df[["y", "school"]]
    sigma_data_df = pandas.concat([data_df[["school", "sigma"]], data_df[["school", "sigma"]]])

    model_string = """
    y' ~ normal(theta[school], sigma[y]);
    theta' = mu + z[school] * tau;
    z ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    """

    with pytest.raises(Exception, match="Subscript y not found in dataframe sigma_data"):
        Model({"y_data": y_data_df, "sigma_data": sigma_data_df}, model_string=model_string)

    model_string = """
    y' ~ normal(theta[school], sigma[school, rabbit]);
    theta' = mu + z[school] * tau;
    z ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    """

    with pytest.raises(Exception, match="Subscript rabbit not found in dataframe y_data"):
        Model({"y_data": y_data_df, "sigma_data": sigma_data_df}, model_string=model_string)
