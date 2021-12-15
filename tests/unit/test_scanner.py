import pytest
from rat.scanner import *

def test_scaner_invalid():
    input_str = "tau<lower===0.0> ~ normal(0.0, 1.0)"
    parsed = scanner(input_str)
    print()
    for val in parsed:
        print(val.value.ljust(20), val.token_type)

