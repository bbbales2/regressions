import numpy
import pytest
from rat.parser import RatParser
from rat.math import negative_binomial, negative_binomial_log


def test_negative_binomial_parser():
    model_string = "y ~ negative_binomial(mu, sigma);"

    parsed = RatParser().parse(model_string)


def test_negative_binomial_values():
    y = 2
    mu = 0.3
    phi = 2.2

    # stats::dnbinom(2, mu = 0.3, size = 2.2, log = TRUE)
    assert negative_binomial(y, mu, phi) == pytest.approx(-3.263300)

    y = 10
    mu = 11.7
    phi = 3.0

    # stats::dnbinom(10, mu = 11.7, size = 3.0, log = TRUE)
    assert negative_binomial(y, mu, phi) == pytest.approx(-2.860637)


def test_negative_binomial_log_values():
    y = 2
    eta = numpy.log(0.3)
    phi = 2.2

    # stats::dnbinom(2, mu = 0.3, size = 2.2, log = TRUE)
    assert negative_binomial_log(y, eta, phi) == pytest.approx(-3.263300)

    y = 10
    eta = numpy.log(11.7)
    phi = 3.0

    # stats::dnbinom(10, mu = 11.7, size = 3.0, log = TRUE)
    assert negative_binomial_log(y, eta, phi) == pytest.approx(-2.860637)
