import pandas
import pytest
from rat import variables
from rat.scanner import Scanner
from rat.parser import Parser
from rat.compiler import log_normal


def test_log_normal_parser():
    model_string = "y ~ log_normal(mu, sigma);"

    parsed = Parser(Scanner(model_string).scan()[0], ["y", "mu", "sigma"], model_string).statement()

    assert str(parsed) == "LogNormal(Data(y), Data(mu), Data(sigma))"


def test_log_normal_values():
    y = 1.7
    mu = 0.3
    sigma = 2.2

    # stats::dlnorm(1.7, 0.3, 2.2, log = TRUE)
    assert log_normal(y, mu, sigma) == pytest.approx(-2.243519)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
