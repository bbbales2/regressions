import jax.numpy
import pytest

import rat.parser
from rat.parser import Parser, ParseError
from rat.scanner import Scanner
from rat.ops import *


def test_parser_multiple():
    input_str = """
    score_diff ~ normal(skills[home_team, year]-skills[away_team, year],sigma);
    skills[team, year] ~ normal(skills_mu[year], tau);
    tau<lower=0.0 + 5.0> ~ normal(0.0, 1.0);
    sigma<lower=0.0> ~ normal(0.0, 10.0);
    """
    data_names = [
        "game_id",
        "date",
        "home_score",
        "away_score",
        "home_team",
        "away_team",
    ]
    print("FULL MODEL TEST - RUNNING TEST MODEL...")
    scanned_lines = Scanner(input_str).scan()
    for line in scanned_lines:
        print(Parser(line, data_names, line).statement())
    print("END FULL MODEL TEST")


def test_parser_simple_constraint_sampling():
    input_str = "tau<lower=0.0> ~ normal(-2.0 + 1.0, 1.0);"
    data_names = []

    statement = Parser(Scanner(input_str).scan()[0], data_names, input_str).statement()
    expected = Normal(
        Param(
            name="tau",
            subscript=None,
            lower=RealConstant(0.0),
            upper=RealConstant(float("inf")),
        ),
        Sum(PrefixNegation(RealConstant(2.0)), RealConstant(1.0)),
        RealConstant(1.0),
    )
    assert statement.__str__() == expected.__str__()


def test_parser_complex_constraint_sampling():
    input_str = "tau<lower=exp(0.0)> ~ normal(0.0, 1.0);"
    data_names = []
    statement = Parser(Scanner(input_str).scan()[0], data_names, input_str).statement()
    expected = Normal(
        Param(
            name="tau",
            subscript=None,
            lower=Exp(RealConstant(0.0)),
            upper=RealConstant(float("inf")),
        ),
        RealConstant(0.0),
        RealConstant(1.0),
    )
    assert statement.__str__() == expected.__str__()


def test_parser_rhs_index_shift():
    input_str = "tau<lower=0.0> ~ normal(skills_mu[shift(year, 1), team], 1.0);"
    data_names = ["year"]
    statement = Parser(Scanner(input_str).scan()[0], data_names, input_str).statement()
    expected = Normal(
        Param(
            name="tau",
            subscript=None,
            lower=RealConstant(0.0),
            upper=RealConstant(float("inf")),
        ),
        Param(name="skills_mu", subscript=Subscript(names=("year", "team"), shifts=(1, None))),
        RealConstant(1.0),
    )
    assert statement.__str__() == expected.__str__()


def test_parser_rhs_index_shift_multiple():
    input_str = "tau<lower=0.0> ~ normal(skills_mu[shift(year, 1), shift(team, -1)], 1.0);"
    data_names = ["year", "team"]
    print([(x.__class__.__name__, x.value) for x in Scanner(input_str).scan()[0]])
    statement = Parser(Scanner(input_str).scan()[0], data_names, input_str).statement()
    expected = Normal(
        Param(
            name="tau",
            subscript=None,
            lower=RealConstant(0.0),
            upper=RealConstant(float("inf")),
        ),
        Param(name="skills_mu", subscript=Subscript(names=("year", "team"), shifts=(1, -1))),
        RealConstant(1.0),
    )

    assert statement.__str__() == expected.__str__()


def test_parser_invalid_statement():
    with pytest.raises(Exception, match="normal distribution needs 2 parameters"):
        test_string = "tau<lower=0.0> ~ normal(skills_mu[year]);"
        data_names = ["year", "skills_mu"]
        statement = Parser(Scanner(test_string).scan()[0], data_names, test_string).statement()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
