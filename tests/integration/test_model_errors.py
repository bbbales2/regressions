import logging
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_unknown_subscript():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = "y' ~ normal(theta[rabbit], sigma);"

    with pytest.raises(Exception, match="Subscript rabbit not found in dataframe"):
        model = Model(model_string, data_df)


def test_lines_in_wrong_order_for_primes():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    mu ~ normal(-0.5, 0.3);
    y' ~ bernoulli_logit(mu[group]);
    """

    with pytest.raises(Exception, match="The primed uses must come last"):
        model = Model(model_string=model_string, data=data_df)


def test_lines_in_wrong_order_for_assignments():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    y' ~ bernoulli_logit(mu[group]);
    mu[group]' = mu2[group];
    mu ~ normal(-0.5, 0.3);
    mu2 ~ normal(-0.5, 0.3);
    """

    with pytest.raises(Exception, match="A variable cannot be used after it is assigned"):
        model = Model(model_string=model_string, data=data_df)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
