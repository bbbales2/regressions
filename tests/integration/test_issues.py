import os
import pathlib
import pandas
import pytest

from rat.model import Model

test_dir = pathlib.Path(__file__).parent

# https://github.com/bbbales2/regressions/issues/88
def test_distribution_lhs_expression_issue_88():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = """
    2.0 * y' - y ~ normal(theta[school], sigma);
    theta' = mu + z[school] * tau;
    z ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    """

    model = Model(data_df, model_string)
    fit = model.optimize()

    mu_df = fit.draws("mu")
    tau_df = fit.draws("tau")

    assert mu_df["mu"][0] == pytest.approx(4.61934000, rel=1e-2)
    assert tau_df["tau"][0] == pytest.approx(0.36975800, rel=1e-2)

#https://github.com/bbbales2/regressions/issues/89
def test_primed_variable_does_not_exist_issue_89():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    made' ~ normal(mu[group], 1.0);
    """

    with pytest.raises(Exception, match="primed variable does not exist"):
        model = Model(data_df, model_string=model_string)