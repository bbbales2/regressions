import logging
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_bernoulli():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    y[n]' ~ bernoulli_logit(mu[group[n]]);
    mu[group] ~ normal(-0.5, 0.3);
    """

    model = Model(data_df, model_string=model_string)
    fit = model.optimize(init=0.1)
    mu_df = fit.draws("mu")

    # This is ordered from group 1 to 5 -- assumes draws are too
    mu_ref = [0.0797716, -0.4084060, -0.8545570, -0.9920960, -0.7676070]
    assert mu_df["mu"].to_list() == pytest.approx(mu_ref, rel=1e-2)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
