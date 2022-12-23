import os
import pathlib
import pandas
import pytest

from rat.model import Model
from rat import optimize

test_dir = pathlib.Path(__file__).parent
issue_data_dir = test_dir / "issue_data"

# https://github.com/bbbales2/regressions/issues/84
def test_unkown_unary_function_issue_84():
    data_df = pandas.DataFrame({
        "school" : [1, 2, 3, 4],
        "offset" : [10.0, 0.0, 0.0, 0.0]
    })

    model_string = """
    offset[school]' ~ normal(mu[school], 1000.0);
    mu[school]' ~ normal(
        reasfdsdaaltasdfsadasfdfsd(0.7) + 1.0,
        0.1
    );
    """

    with pytest.raises(Exception, match="Error calling unknown function reasfdsdaaltasdfsadasfdfsd"):
        Model(model_string, data_df)


# https://github.com/bbbales2/regressions/issues/90
def test_ifelse_issue_90():
    data_df = pandas.read_csv(issue_data_dir / "90.csv")

    with open(issue_data_dir / "90.rat") as f:
        model_string = f.read()

    Model(model_string, data_df)


# https://github.com/bbbales2/regressions/issues/88
def test_distribution_lhs_expression_issue_88():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = """
    (2.0 * y[school]' - y[school]) ~ normal(theta[school], sigma[school]);
    theta[school]' = mu + z[school] * tau;
    z[school] ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    """

    model = Model(model_string, data_df)
    fit = optimize(model)

    mu_df = fit.draws("mu")
    tau_df = fit.draws("tau")

    assert mu_df["mu"][0] == pytest.approx(4.61934000, rel=1e-2)
    assert tau_df["tau"][0] == pytest.approx(0.36975800, rel=1e-2)


# https://github.com/bbbales2/regressions/issues/89
def test_primed_variable_does_not_exist_issue_89():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    made' ~ normal(mu[group], 1.0);
    """

    with pytest.raises(Exception, match="group is in control flow/subscripts and must be a primary variable subscript, but it is not"):
        model = Model(model_string=model_string, data=data_df)
