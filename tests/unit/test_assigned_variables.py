import pytest
from rat.parser import Parser
from rat.scanner import scanner
from rat.compiler import compile
import pandas as pd

def test_assignment_compiler():
    input_str = """
    x ~ normal(mean[year], a);
    mean[year] ~ normal(0, 1);
    a = 1;
    b ~ normal(0.0, 1.0);
    """
    data_names = [
        "x",
        "year"
    ]

    test_df = pd.DataFrame({"x": [x for x in range(10)], "year": [x for x in range(2010, 2020)]})
    print("FULL MODEL TEST - RUNNING TEST MODEL...")
    print(test_df)
    print("*" * 10)
    parsed_lines = []
    for line in input_str.split("\n"):
        line = line.lstrip().rstrip()
        if not line:
            continue
        #print(f"----running for line [{line}]")
        #print(line)
        #print("{", [(x.__class__.__name__, x.value) for x in scanner(line)], "}")
        parsed = Parser(scanner(line), data_names).statement()
        #print(parsed)
        parsed_lines.append(parsed)
    print("END FULL MODEL TEST")
    data_variables, parameter_variables, index_variables, assigned_parameter_variables, line_functions = compile(test_df, parsed_lines)
    for f in line_functions:
        pass
        #print(f.code())