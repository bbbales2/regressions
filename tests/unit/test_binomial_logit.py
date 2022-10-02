import numpy
import pytest
from rat.parser import RatParser
from rat.math import binomial_logit


def test_binomial_logit_parser():
    model_string = "y ~ binomial_logit(N, mu);"

    parsed = RatParser().parse(model_string)


def test_binomial_logit_values():
    y = 2
    N = 10
    mu = 3.7

    # stats::dbinom(2, size = 10, prob = 1 / (1 + exp(-3.7)), log = TRUE)
    assert binomial_logit(y, N, mu) == pytest.approx(-26.037566)

    y = 10
    N = 12
    mu = 3.0

    # stats::dbinom(10, size = 12, prob = 1 / (1 + exp(-3.0)), log = TRUE)
    assert binomial_logit(y, N, mu) == pytest.approx(-2.393393)
