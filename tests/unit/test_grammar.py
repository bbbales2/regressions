from __future__ import generator_stop

import json
import os
import pathlib
from pprint import pprint

import tatsu
from tatsu.ast import AST
from tatsu.walkers import NodeWalker
from rat.grammar import grammar

test_dir = pathlib.Path(__file__).parent


def test_grammar():
    with open(os.path.join(test_dir, "grammar.rat")) as f:
        text = f.read()

    parser = tatsu.compile(grammar())
    ast = parser.parse(text, trace=True, colorize=True)

    print()
    print("# SIMPLE PARSE")
    print("# AST")
    pprint(ast, width=20, indent=4)
    print()

    print("# JSON")
    print(json.dumps(ast, indent=4))
    print()


if __name__ == "__main__":
    test_grammar()
