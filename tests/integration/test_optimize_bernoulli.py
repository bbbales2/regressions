import logging
import os
import pathlib
import pandas
import pytest
import rat

test_dir = pathlib.Path(__file__).parent


def test_optimize_bernoulli():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    y[n]' ~ bernoulli_logit(mu[group[n]]);
    mu[group] ~ normal(-0.5, 0.3);
    """

    model = rat.Model(model_string=model_string, data=data_df)
    fit = rat.optimize(model, init=0.1)
    mu_df = fit.draws("mu")

    # This is ordered from group 1 to 5 -- assumes draws are too
    mu_ref = [0.0797716, -0.4084060, -0.8545570, -0.9920960, -0.7676070]
    assert mu_df["mu"].to_list() == pytest.approx(mu_ref, rel=1e-2)


def test_expr_bernoulli():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    y[n]' ~ bernoulli_logit(mu[group[n]]);
    mu[group] ~ normal(-0.5, 0.3);
    """

    model = rat.Model(model_string=model_string, data=data_df)
    fit = rat.optimize(model, init=0.1)
    mu_df = fit.expr("mu[group]")

    # This is ordered from group 1 to 5 -- assumes draws are too
    mu_ref = [0.0797716, -0.4084060, -0.8545570, -0.9920960, -0.7676070]
    for chain, per_chain_mu_df in mu_df.groupby("chain"):
        assert per_chain_mu_df["value"].to_list() == pytest.approx(mu_ref, rel=1e-2)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
