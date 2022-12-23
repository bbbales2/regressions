from dataclasses import dataclass, field
from typing import List

from . import ast
from tatsu.model import NodeWalker


class RatWalker(NodeWalker):
    def walk_Program(self, node: ast.Program):
        for statement in node.statements:
            self.walk(statement)

    def walk_Statement(self, node: ast.Statement):
        self.walk(node.left)
        self.walk(node.right)

    def walk_Binary(self, node: ast.Binary):
        self.walk(node.left)
        self.walk(node.right)

    def walk_IfElse(self, node: ast.IfElse):
        self.walk(node.predicate)
        self.walk(node.left)
        self.walk(node.right)

    def walk_FunctionCall(self, node: ast.FunctionCall):
        if node.arglist:
            for arg in node.arglist:
                self.walk(arg)

    def walk_Variable(self, node: ast.Variable):
        if node.constraints:
            self.walk(node.constraints.left)
            self.walk(node.constraints.right)

        if node.arglist:
            for arg in node.arglist:
                self.walk(arg)

    def walk_Constraints(self, node: ast.Constraints):
        self.walk(node.left)
        if node.right:
            self.walk(node.right)


@dataclass
class AstFlattener(NodeWalker):
    nodes: List[ast.ModelBase] = field(default_factory=list)

    def walk_Program(self, node: ast.Program):
        self.nodes.append(node)
        for statement in node.statements:
            self.walk(statement)

    def walk_Statement(self, node: ast.Statement):
        self.nodes.append(node)
        self.walk(node.left)
        self.walk(node.right)

    def walk_Binary(self, node: ast.Binary):
        self.nodes.append(node)
        self.walk(node.left)
        self.walk(node.right)

    def walk_IfElse(self, node: ast.IfElse):
        self.nodes.append(node)
        self.walk(node.predicate)
        self.walk(node.left)
        self.walk(node.right)

    def walk_FunctionCall(self, node: ast.FunctionCall):
        self.nodes.append(node)
        if node.arglist:
            for arg in node.arglist:
                self.walk(arg)

    def walk_Variable(self, node: ast.Variable):
        self.nodes.append(node)
        if node.constraints:
            self.walk(node.constraints)

        if node.arglist:
            for arg in node.arglist:
                self.walk(arg)

    def walk_Constraints(self, node: ast.Constraints):
        self.walk(node.left)
        if node.right:
            self.walk(node.right)


def flatten_ast(node: ast.ModelBase) -> List[ast.ModelBase]:
    ast_flattener = AstFlattener()
    ast_flattener.walk(node)
    return ast_flattener.nodes
