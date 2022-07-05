from __future__ import generator_stop

import json
import os
import pathlib
from pprint import pprint
from dataclasses import dataclass, field

from typing import List
import tatsu
from tatsu.util import asjson
from tatsu.ast import AST
from tatsu.walkers import NodeWalker
from tatsu.model import ModelBuilderSemantics
from rat.grammar import grammar
from rat.parser2 import RatParser
from rat.walker import RatWalker
import rat.ast2 as ast2

test_dir = pathlib.Path(__file__).parent

class VariableWalker(RatWalker):
    def walk_Variable(self, node : ast2.Variable):
        print(node.name)


def get_primary_symbol_key(statement : ast2.Statement):
    @dataclass
    class PrimaryWalker(RatWalker):
        marked : ast2.Variable = None
        candidates : List[ast2.Variable] = field(default_factory=list)

        def walk_Variable(self, node: ast2.Variable):
            if node.prime:
                if self.marked == None:
                    self.marked = node
                else:
                    raise Exception("etc. etc.1")
            else:
                self.candidates.append(node)
    
    walker = PrimaryWalker()
    walker.walk(statement)
    marked = walker.marked
    candidates = walker.candidates

    if marked != None:
        return marked.name
    
    if len(candidates) == 1:
        return candidates[0].name
    
    if len(candidates) > 1:
        raise Exception("etc. etc.2")

    if len(candidates) == 0:
        raise Exception("etc. etc.3")

def discover_subscript_names(program : ast2.Program):
    @dataclass
    class PrimaryWalker(RatWalker):
        def walk_Variable(self, node: ast2.Variable):
            if node.name == self.primary_name:
                print("ensure: ", node.name, node.arglist)
            else:
                print("check: ", node.name, node.arglist)

    for statement in program.ast:
        primary_name = get_primary_symbol_key(statement)

        walker = PrimaryWalker(primary_name)
        walker.walk(program)
    
    for statement in program.ast:
        primary_name = get_primary_symbol_key(statement)

        walker = PrimaryWalker(primary_name)
        walker.walk(program)




def test_grammar():
    with open(os.path.join(test_dir, "grammar.rat")) as f:
        text = f.read()

    semantics = ModelBuilderSemantics()
    parser = RatParser(semantics = semantics)
    program = (lambda : parser.parse(text))()

    #for statement in program.ast:
    #    print(get_primary_symbol_key(statement))

    parser = tatsu.compile(grammar())
    ast = parser.parse(text, trace=True, colorize=True)

    print()
    print("# SIMPLE PARSE")
    print("# AST")
    pprint(ast, width=20, indent=4)
    print()

    print("# JSON")
    print(json.dumps(asjson(ast), indent=4))
    print()


if __name__ == "__main__":
    test_grammar()