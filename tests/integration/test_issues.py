import os
import pathlib
import pandas
import pytest

from rat.model import Model
from rat import optimize

test_dir = pathlib.Path(__file__).parent
issue_data_dir = test_dir / "issue_data"

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
    (2.0 * y' - y) ~ normal(theta[school], sigma);
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


a = {
    "statements": [
        {
            "left": {
                "left": {
                    "value": "2.0",
                    "parseinfo": {"tokenizer": None, "rule": "literal", "pos": 5, "endpos": 8, "line": 1, "endline": 1, "alerts": []},
                },
                "op": "*",
                "right": {
                    "left": {
                        "prime": "'",
                        "arglist": None,
                        "constraints": None,
                        "name": "y",
                        "parseinfo": {
                            "tokenizer": None,
                            "rule": "variable",
                            "pos": 11,
                            "endpos": 13,
                            "line": 1,
                            "endline": 1,
                            "alerts": [],
                        },
                    },
                    "op": "-",
                    "right": {
                        "prime": None,
                        "arglist": None,
                        "constraints": None,
                        "name": "y",
                        "parseinfo": {
                            "tokenizer": None,
                            "rule": "variable",
                            "pos": 16,
                            "endpos": 17,
                            "line": 1,
                            "endline": 1,
                            "alerts": [],
                        },
                    },
                    "parseinfo": {"tokenizer": None, "rule": "addition", "pos": 11, "endpos": 17, "line": 1, "endline": 1, "alerts": []},
                },
                "parseinfo": {"tokenizer": None, "rule": "multiplication", "pos": 5, "endpos": 17, "line": 1, "endline": 1, "alerts": []},
            },
            "op": "~",
            "right": {
                "arglist": [
                    {
                        "prime": None,
                        "arglist": [
                            {
                                "prime": None,
                                "arglist": None,
                                "constraints": None,
                                "name": "school",
                                "parseinfo": {
                                    "tokenizer": None,
                                    "rule": "variable",
                                    "pos": 33,
                                    "endpos": 39,
                                    "line": 1,
                                    "endline": 1,
                                    "alerts": [],
                                },
                            }
                        ],
                        "constraints": None,
                        "name": "theta",
                        "parseinfo": {
                            "tokenizer": None,
                            "rule": "variable",
                            "pos": 27,
                            "endpos": 40,
                            "line": 1,
                            "endline": 1,
                            "alerts": [],
                        },
                    },
                    {
                        "prime": None,
                        "arglist": None,
                        "constraints": None,
                        "name": "sigma",
                        "parseinfo": {
                            "tokenizer": None,
                            "rule": "variable",
                            "pos": 42,
                            "endpos": 47,
                            "line": 1,
                            "endline": 1,
                            "alerts": [],
                        },
                    },
                ],
                "name": "normal",
                "parseinfo": {"tokenizer": None, "rule": "function_call", "pos": 20, "endpos": 48, "line": 1, "endline": 1, "alerts": []},
            },
            "parseinfo": {"tokenizer": None, "rule": "sample_statement", "pos": 5, "endpos": 49, "line": 1, "endline": 1, "alerts": []},
        },
    ],
    "parseinfo": {"tokenizer": None, "rule": "start", "pos": 0, "endpos": 191, "line": 0, "endline": 7, "alerts": []},
}
