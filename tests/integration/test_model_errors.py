import logging
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_missing_prior_or_assign():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    parsed_lines = [
        ast.Normal(
            ast.Data("y"),
            ast.Param("theta", ast.Subscript(names=(ast.SubscriptColumn("school"),), shifts=(ast.IntegerConstant(0),))),
            ast.Data("sigma"),
        ),
    ]

    with pytest.raises(Exception):
        model = Model(data_df, parsed_lines)


def test_optimize_unknown_subscript():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    parsed_lines = [
        ast.Normal(
            ast.Data("y"),
            ast.Param("theta", ast.Subscript(names=(ast.SubscriptColumn("rabbit"),), shifts=(ast.IntegerConstant(0),))),
            ast.Data("sigma"),
        ),
    ]

    with pytest.raises(Exception):
        model = Model(data_df, parsed_lines)


def test_optimize_missing_subscript():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    parsed_lines = [
        ast.Normal(
            ast.Data("y"),
            ast.Param("theta", ast.Subscript(names=(ast.SubscriptColumn("school"),), shifts=(ast.IntegerConstant(0),))),
            ast.Data("sigma"),
        ),
        ast.Normal(ast.Param("theta"), ast.RealConstant(0.0), ast.RealConstant(5.0)),
    ]

    with pytest.raises(Exception):
        model = Model(data_df, parsed_lines)


def test_lines_in_wrong_order_for_primes():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    mu ~ normal(-0.5, 0.3);
    y' ~ bernoulli_logit(mu[group]);
    """

    with pytest.raises(Exception, match="The primed uses must come last"):
        model = Model(data_df, model_string=model_string)


def test_lines_in_wrong_order_for_assignments():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    y' ~ bernoulli_logit(mu[group]);
    mu[group]' = 0.1 + mu2[group]; # If you remove the '0.1 +' this fails to parse
    mu ~ normal(-0.5, 0.3);
    mu2 ~ normal(-0.5, 0.3);
    """

    with pytest.raises(Exception, match="A variable cannot be used after it is assigned"):
        model = Model(data_df, model_string=model_string)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
