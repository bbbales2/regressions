import logging
import math
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_ifelse_basic():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = """
    mu[school] ~ normal(ifelse(school' <= 4, 0.0, 1.0), 0.1);
    """

    model = Model(data_df, model_string)
    fit = model.optimize()

    mu_df = fit.draws("mu")

    for row in mu_df.itertuples():
       assert row.mu == pytest.approx(0.0 if row.school <= 4 else 1.0, abs=1e-3, rel=1e-3)

def test_ifelse_recursive():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    data_df = pandas.DataFrame({
        "school" : [1, 2, 3, 4],
        "offset" : [10.0, 0.0, 0.0, 0.0]
    })

    model_string = """
    # This first line more or less shouldn't matter
    offset' ~ normal(mu[school], 1000.0);
    mu[school]' ~ normal(
        ifelse(
            school == 1,
            10.0,
            real(mu[shift(school, 1)]) + 1.0
        ),
        0.1
    );
    """

    model = Model(data_df, model_string)
    fit = model.optimize()

    mu_df = fit.draws("mu")

    for row in mu_df.itertuples():
       assert row.mu == pytest.approx(9.0 + row.school, abs=1e-3, rel=1e-3)

def test_ifelse_recursive_assignment():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    data_df = pandas.DataFrame({
        "school" : [1, 2, 3, 4],
        "offset" : [10.0, 0.0, 0.0, 0.0]
    })

    model_string = """
    # This first line more or less shouldn't matter
    offset' ~ normal(mu[school], 1000.0);
    mu[school]' = ifelse(
        school[school] == 1,
        10.0,
        mu[shift(school, 1)] + 1.0
    ) + epsilon[school];
    epsilon[school] ~ normal(0.0, 0.1);
    """

    model = Model(data_df, model_string)
    fit = model.optimize()

    mu_df = fit.draws("mu")

    for row in mu_df.itertuples():
       assert row.mu == pytest.approx(9.0 + row.school, abs=1e-3, rel=1e-3)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
