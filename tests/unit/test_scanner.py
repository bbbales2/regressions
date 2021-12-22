import pytest
from rat.scanner import *


def test_scaner_invalid():
    input_str = "tau<lower===0.0> ~ normal(0.0, 1.0);"
    parsed = Scanner(input_str)
    print()
    for val in parsed.scan()[0]:
        print(val.value.ljust(20), val.token_type)


def test_scanner_valid_1():
    input_str = "tau<lower=0.0> ~ normal(-2.0 - 1.0, 1.0);"
    parsed = Scanner(input_str)
    print()
    for val in parsed.scan()[0]:
        print(val.value.ljust(20), val.token_type)


def test_scanner_valid_multiline_1():
    input_str = """
    y ~ bernoulli_logit(
      a +
      b
    );
    """
    parsed = Scanner(input_str)
    print()
    for line in parsed.scan():
        for token in line:
            print(token.value.ljust(20), token.token_type)
