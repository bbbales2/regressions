import pytest
from rat.scanner import *


def test_scanner_invalid_scientific():
    input_str = "g<lower=1.4ef> ~ normal(0.0, 1.0);"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError):
        parsed.scan()


def test_scanner_invalid_identifier():
    input_str = "$$1337 ~ bernoulli_logit(p);"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError):
        parsed.scan()


def test_scanner_invalid_identifier_2():
    input_str = "abc_! ~ cauchy(0.0, 1.0)"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError):
        parsed.scan()


def test_scanner_missing_terminate():
    input_str = "mu ~ bernoulli_logit(p)"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError):
        parsed.scan()


def test_scanner_valid_1():
    input_str = "tau<lower=0.0> ~ normal(-2.0 - 1.0, 1.0);"
    #            ^        ^         ^         ^
    #            1        10        20        30
    expected = [
        Identifier("tau", 0, 1),
        Operator("<", 0, 4),
        Identifier("lower", 0, 5),
        Operator("=", 0, 10),
        RealLiteral("0.0", 0, 11),
        Operator(">", 0, 14),
        Special("~", 0, 16),
        Identifier("normal", 0, 18),
        Special("(", 0, 24),
        Operator("-", 0, 25),
        RealLiteral("2.0", 0, 26),
        Operator("-", 0, 30),
        RealLiteral("1.0", 0, 32),
        Special(",", 0, 35),
        RealLiteral("1.0", 0, 37),
        Special(")", 0, 40),
        Terminate(";", 0, 41),
    ]
    parsed = Scanner(input_str)
    for index, val in enumerate(parsed.scan()[0]):
        # print(val.value.ljust(20), val.token_type)
        true_token = expected[index]
        assert true_token.value == val.value
        assert true_token.line_index == val.line_index
        assert true_token.column_index == val.column_index


def test_scanner_complex():
    input_str = "tau<lower=1.4e5*2,upper=1.1>~ normal(exp(-2.0) - 1.0*a+1.0e1-a_412+1, 1.0);"
    #            ^        ^         ^         ^         ^         ^         ^         ^
    #            1        10        20        30        40        50        60        70
    expected = [
        Identifier("tau", 0, 1),
        Operator("<", 0, 4),
        Identifier("lower", 0, 5),
        Operator("=", 0, 10),
        RealLiteral("1.4e5", 0, 11),
        Operator("*", 0, 16),
        IntLiteral("2", 0, 17),
        Special(",", 0, 18),
        Identifier("upper", 0, 19),
        Operator("=", 0, 24),
        RealLiteral("1.1", 0, 25),
        Operator(">", 0, 28),
        Special("~", 0, 29),
        Identifier("normal", 0, 31),
        Special("(", 0, 37),
        Identifier("exp", 0, 38),
        Special("(", 0, 41),
        Operator("-", 0, 42),
        RealLiteral("2.0", 0, 43),
        Special(")", 0, 46),
        Operator("-", 0, 48),
        RealLiteral("1.0", 0, 50),
        Operator("*", 0, 53),
        Identifier("a", 0, 54),
        Operator("+", 0, 55),
        RealLiteral("1.0e1", 0, 56),
        Operator("-", 0, 61),
        Identifier("a_412", 0, 62),
        Operator("+", 0, 67),
        IntLiteral("1", 0, 68),
        Special(",", 0, 69),
        RealLiteral("1.0", 0, 71),
        Special(")", 0, 74),
        Terminate(";", 0, 75),
    ]
    parsed = Scanner(input_str)
    for index, val in enumerate(parsed.scan()[0]):
        # print(val.value.ljust(20), val.token_type)
        true_token = expected[index]
        assert true_token.value == val.value
        assert true_token.line_index == val.line_index
        assert true_token.column_index == val.column_index


def test_scanner_valid_multiline_1():
    input_str = """
    y ~ bernoulli_logit(
      a +
      b
    );
    """
    parsed = Scanner(input_str)
    lines = parsed.scan()
    assert len(lines) == 1
    assert len(lines[0]) == 8
    print()
    for line in lines:
        for token in line:
            print(token.value.ljust(20), token.token_type)


def test_scanner_valid_multiline_2():
    input_str = """
y ~ bernoulli_logit(
a +
b
);a=
1;
"""
    parsed = Scanner(input_str)
    lines = parsed.scan()
    assert len(lines) == 2
    assert lines[0][0].value == "y"
    assert lines[0][0].line_index == 1
    assert lines[1][0].value == "a"
    assert lines[1][0].line_index == 4
