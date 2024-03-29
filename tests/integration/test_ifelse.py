import logging
import math
import os
import pathlib
import pandas
import pytest
import rat

from rat.model import Model
from rat.optimizer import optimize

test_dir = pathlib.Path(__file__).parent


def test_ifelse_basic():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = """
    y[school]' ~ normal(ifelse(school <= 4, mu1, mu2), 0.1);
    """

    model = Model(model_string, data_df)
    fit = optimize(model)

    mu1_df = fit.draws("mu1")
    mu2_df = fit.draws("mu2")

    assert mu1_df.loc[0].mu1 == pytest.approx(10.0)
    assert mu2_df.loc[0].mu2 == pytest.approx(7.5)


# def test_ifelse_recursive():
#     data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))
#
#     data_df = pandas.DataFrame({"school": [1, 2, 3, 4], "offset": [10.0, 0.0, 0.0, 0.0]})
#
#     model_string = """
#     # This first line more or less shouldn't matter
#     offset' ~ normal(mu[school], 1000.0);
#     mu[school]' ~ normal(
#         ifelse(
#             school == 1,
#             10.0,
#             real(mu[shift(school, 1)]) + 1.0
#         ),
#         0.1
#     );
#     """
#
#     model = Model(model_string, data_df)
#     fit = rat.optimize(model)
#
#     mu_df = fit.draws("mu")
#
#     for row in mu_df.itertuples():
#         assert row.mu == pytest.approx(9.0 + row.school, abs=1e-3, rel=1e-3)
#
#
# def test_ifelse_recursive_assignment():
#     data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))
#
#     data_df = pandas.DataFrame({"school": [1, 2, 3, 4], "offset": [10.0, 0.0, 0.0, 0.0]})
#
#     model_string = """
#     # This first line more or less shouldn't matter
#     offset' ~ normal(mu[school], 1000.0);
#     mu[school]' = ifelse(
#         school == 1,
#         10.0,
#         mu[shift(school, 1)] + 1.0
#     ) + epsilon[school];
#     epsilon[school] ~ normal(0.0, 0.1);
#     """
#
#     model = Model(model_string, data_df)
#     fit = rat.optimize(model)
#
#     mu_df = fit.draws("mu")
#
#     for row in mu_df.itertuples():
#         assert row.mu == pytest.approx(9.0 + row.school, abs=1e-3, rel=1e-3)
#
#
# def test_ifelse_recursive_assignment2():
#     # this is a test for ifelse when the length of a subscript parameter is different from the # of elements in
#     # the subscript data column
#     data_df = pandas.DataFrame({"y": [1.0, 2.0, 3.0, 2.0], "group": [1, 2, 3, 2]})
#
#     model_string = """
#         # This first line more or less shouldn't matter
#         y' ~ normal(mu[group], 1.0);
#         mu[group]' = ifelse(
#             group == 1,
#             1.0,
#             mu[shift(group, 1)] + 1.0
#         ) + epsilon[group];
#         epsilon[group] ~ normal(0.0, 0.1);
#         """
#
#     model = Model(model_string, data_df)
#     fit = rat.optimize(model)
#
#     mu_df = fit.draws("mu")
#     for row in mu_df.itertuples():
#         assert row.mu == pytest.approx(row.group, abs=1e-3, rel=1e-3)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
