import pytest
from rat.parser import Parser
from rat.scanner import scanner
from rat.ops import *


def test_parser_multiple():
    input_str = """
    score_diff~normal(skills[home_team, year]-skills[away_team, year],sigma);
    skills[union(home_team, away_team), year] ~ normal(skills_mu[year], tau);
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
    for line in input_str.split("\n"):
        line = line.lstrip().rstrip()
        if not line:
            continue
        print(f"----running for line [{line}]")
        print(line)
        print("{", [(x.__class__.__name__, x.value) for x in scanner(line)], "}")
        print(Parser(scanner(line), data_names).statement())
    print("END FULL MODEL TEST")


def test_parser_simple_constraint_sampling():
    input_str = "tau<lower=0.0> ~ normal(0.0, 1.0);"
    data_names = []
    statement = Parser(scanner(input_str), data_names).statement()
    expected = Normal(
        Param(
            name="tau",
            index=None,
            lower=RealConstant(0.0),
            upper=RealConstant(float("inf")),
        ),
        RealConstant(0.0),
        RealConstant(1.0),
    )
    assert statement.__str__() == expected.__str__()


def test_parser_complex_constraint_sampling():
    input_str = "tau<lower=exp(0.0) ^ 1> ~ normal(0.0, 1.0);"
    data_names = []
    statement = Parser(scanner(input_str), data_names).statement()
    expected = Normal(
        Param(
            name="tau",
            index=None,
            lower=Pow(Exp(RealConstant(0.0)), IntegerConstant(1)),
            upper=RealConstant(float("inf")),
        ),
        RealConstant(0.0),
        RealConstant(1.0),
    )
    assert statement.__str__() == expected.__str__()


def test_parser_rhs_index_shift():
    input_str = "tau<lower=0.0> ~ normal(skills_mu[shift(year, 1), team], 1.0);"
    data_names = ["year"]
    statement = Parser(scanner(input_str), data_names).statement()
    expected = Normal(
        Param(
            name="tau",
            index=None,
            lower=RealConstant(0.0),
            upper=RealConstant(float("inf")),
        ),
        Param(name="skills_mu", index=Index(names=("year", "team"), shifts=(1, None))),
        RealConstant(1.0),
    )
    assert statement.__str__() == expected.__str__()


def test_parser_rhs_index_shift_multiple():
    input_str = "tau<lower=0.0> ~ normal(skills_mu[shift(year, 1), shift(team, -1)], 1.0);"
    data_names = ["year", "team"]
    print([(x.__class__.__name__, x.value) for x in scanner(input_str)])
    statement = Parser(scanner(input_str), data_names).statement()
    expected = Normal(
        Param(
            name="tau",
            index=None,
            lower=RealConstant(0.0),
            upper=RealConstant(float("inf")),
        ),
        Param(name="skills_mu", index=Index(names=("year", "team"), shifts=(1, -1))),
        RealConstant(1.0),
    )

    assert statement.__str__() == expected.__str__()


def test_parser_lhs_subscript_union():
    input_str = "tau<lower = 0.0>[union(home_team, away_team), year] ~ normal(0.0, 1.0);"
    data_names = ["home_team", "away_team", "year"]
    statement = Parser(scanner(input_str), data_names).statement()

    expected = Normal(
        Param(name="tau", index=Index((("home_team", "away_team"), "year"), shifts=(None, None)), lower=RealConstant("0.0")),
        RealConstant("0.0"),
        RealConstant("1.0"),
    )
    assert statement.__str__() == expected.__str__()
