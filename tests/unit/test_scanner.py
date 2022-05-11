import pytest
from rat.scanner import *


def test_scanner_invalid_scientific():
    input_str = "g<lower=1.4ef> ~ normal(0.0, 1.0);"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError, match="scientific notation"):
        parsed.scan()


def test_scanner_invalid_identifier():
    input_str = "$$1337 ~ bernoulli_logit(p);"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError, match="$"):
        parsed.scan()


def test_scanner_invalid_identifier_2():
    input_str = "abc_! ~ cauchy(0.0, 1.0)"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError, match="!"):
        parsed.scan()


def test_scanner_missing_terminate():
    input_str = "mu ~ bernoulli_logit(p)"
    parsed = Scanner(input_str)
    with pytest.raises(TokenizeError, match="Missing termination character"):
        parsed.scan()


def test_scanner_valid_1():
    input_str = "tau<lower=0.0> ~ normal(-2.0 - 1.0, 1.0);"
    #            ^        ^       ^           ^
    #            0        9       17          24
    expected = [
        Identifier("tau", Position(0, 0, input_str)),
        Operator("<", Position(0, 3, input_str)),
        Identifier("lower", Position(0, 4, input_str)),
        Operator("=", Position(0, 9, input_str)),
        RealLiteral("0.0", Position(0, 10, input_str)),
        Operator(">", Position(0, 13, input_str)),
        Special("~", Position(0, 15, input_str)),
        Identifier("normal", Position(0, 17, input_str)),
        Special("(", Position(0, 23, input_str)),
        Operator("-", Position(0, 24, input_str)),
        RealLiteral("2.0", Position(0, 25, input_str)),
        Operator("-", Position(0, 29, input_str)),
        RealLiteral("1.0", Position(0, 31, input_str)),
        Special(",", Position(0, 34, input_str)),
        RealLiteral("1.0", Position(0, 36, input_str)),
        Special(")", Position(0, 39, input_str)),
        Terminate(";", Position(0, 40, input_str)),
    ]
    parsed = Scanner(input_str)
    for index, val in enumerate(parsed.scan()[0]):
        # print(val.value.ljust(20), val.token_type)
        true_token = expected[index]
        assert true_token.value == val.value
        assert true_token.start == val.start


def test_scanner_complex():
    input_str = "tau<lower=1.4e5*2,upper=1.1>~ normal(exp(-2.0) - 1.0*a+1.0e1-a_412+1, 1.0);"
    #            ^         ^         ^         ^         ^         ^         ^         ^
    #            0         10        20        30        40        50        60        70
    expected = [
        Identifier("tau", Position(0, 0, input_str)),
        Operator("<", Position(0, 3, input_str)),
        Identifier("lower", Position(0, 4, input_str)),
        Operator("=", Position(0, 9, input_str)),
        RealLiteral("1.4e5", Position(0, 10, input_str)),
        Operator("*", Position(0, 15, input_str)),
        IntLiteral("2", Position(0, 16, input_str)),
        Special(",", Position(0, 17, input_str)),
        Identifier("upper", Position(0, 18, input_str)),
        Operator("=", Position(0, 23, input_str)),
        RealLiteral("1.1", Position(0, 24, input_str)),
        Operator(">", Position(0, 27, input_str)),
        Special("~", Position(0, 28, input_str)),
        Identifier("normal", Position(0, 30, input_str)),
        Special("(", Position(0, 36, input_str)),
        Identifier("exp", Position(0, 37, input_str)),
        Special("(", Position(0, 40, input_str)),
        Operator("-", Position(0, 41, input_str)),
        RealLiteral("2.0", Position(0, 42, input_str)),
        Special(")", Position(0, 45, input_str)),
        Operator("-", Position(0, 47, input_str)),
        RealLiteral("1.0", Position(0, 49, input_str)),
        Operator("*", Position(0, 52, input_str)),
        Identifier("a", Position(0, 53, input_str)),
        Operator("+", Position(0, 54, input_str)),
        RealLiteral("1.0e1", Position(0, 55, input_str)),
        Operator("-", Position(0, 60, input_str)),
        Identifier("a_412", Position(0, 61, input_str)),
        Operator("+", Position(0, 66, input_str)),
        IntLiteral("1", Position(0, 67, input_str)),
        Special(",", Position(0, 68, input_str)),
        RealLiteral("1.0", Position(0, 70, input_str)),
        Special(")", Position(0, 73, input_str)),
        Terminate(";", Position(0, 74, input_str)),
    ]
    parsed = Scanner(input_str)
    for index, val in enumerate(parsed.scan()[0]):
        # print(val.value.ljust(20), val.token_type)
        true_token = expected[index]
        assert true_token.value == val.value
        assert true_token.start == val.start


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
    assert len(lines[0]) == 9
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
    assert lines[0][0].start.line == 1
    assert lines[1][0].value == "a"
    assert lines[1][0].start.line == 4


def test_scanner_comment_multiline():
    input_str = """
# this is a comment line
y ~ bernoulli_logit(
a +  # this is a comment in the middle
b# this is a comment without a space
);a=
1;
"""
    parsed = Scanner(input_str)
    lines = parsed.scan()
    assert len(lines) == 2
    assert lines[0][0].value == "y"
    assert lines[0][0].start.line == 2
    assert lines[0][6].value == "b"
    assert lines[1][0].value == "a"
    assert lines[1][0].start.line == 5
