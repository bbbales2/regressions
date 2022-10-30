import pandas
import pytest
from rat.parser import RatParser
from rat.math import log_normal


def test_log_normal_parser():
    model_string = "y ~ log_normal(mu, sigma');"

    parser = RatParser()
    program = (lambda: parser.parse(model_string))()
    assert program.statements[0].right.name == "log_normal"


def test_log_normal_values():
    y = 1.7
    mu = 0.3
    sigma = 2.2

    # stats::dlnorm(1.7, 0.3, 2.2, log = TRUE)
    assert log_normal(y, mu, sigma) == pytest.approx(-2.243519)
