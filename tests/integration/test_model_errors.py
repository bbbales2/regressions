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

    model_string = "y[school]' ~ normal(theta[rabbit], sigma[school]);"

    with pytest.raises(Exception, match="rabbit is in control flow/subscripts and must be a primary variable subscript, but it is not"):
        model = Model(model_string, data_df)


def test_subscripts_not_specified():
    data_df = pandas.read_csv(os.path.join(test_dir, "bernoulli.csv"))

    model_string = """
    y[n]' ~ bernoulli_logit(mu[group[n]]);
    mu[group]' = mu2[group];
    mu ~ normal(-0.5, 0.3);
    mu2 ~ normal(-0.5, 0.3);
    """

    # remember it's regex, so we need to escape parentheses
    with pytest.raises(Exception, match="mu should have 1 subscript\(s\)"):
        model = Model(model_string=model_string, data=data_df)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
