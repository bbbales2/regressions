import pytest
from rat.parser import Parser
from rat.scanner import scanner


def test_parser_multiple():
    input_str = """
    score_diff~normal(skills[home_team, year]-skills[away_team, year],sigma);
    skills[team, year] ~ normal(skills_mu[year], tau);
    tau<lower=0.0 + 5.0> ~ normal(0.0, 1.0);
    sigma<lower=0.0> ~ normal(0.0, 10.0);
    """
    data_names = ["game_id", "date", "home_score", "away_score", "home_team", "away_team"]
    print()
    for line in input_str.split("\n"):
        line = line.lstrip().rstrip()
        if not line: continue
        print(f"running for line [{line}]")
        print(Parser(scanner(line), data_names).statement())


def test_parser_simple_constraint_sampling():
    input_str = "tau<lower=0.0> ~ normal(0.0, 1.0);"
    data_names = []
    statement = Parser(scanner(input_str), data_names).statement()
    print(statement)
    assert statement.__str__() == "Normal(Param(tau, None, lower=RealConstant(0.0), upper=RealConstant(inf)), RealConstant(0.0), RealConstant(1.0))"

def test_parser_complex_constraint_sampling():
    input_str = "tau<lower=exp(0.0) ^ 1> ~ normal(0.0, 1.0);"
    data_names = []
    statement = Parser(scanner(input_str), data_names).statement()
    print(statement)
    assert statement.__str__() == "Normal(Param(tau, None, lower=Pow(Exp(RealConstant(0.0)), IntegerConstant(1)), upper=RealConstant(inf)), RealConstant(0.0), RealConstant(1.0))"