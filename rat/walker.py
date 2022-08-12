from . import ast
from tatsu.model import NodeWalker
from tatsu.model import ModelBuilderSemantics


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
        for arg in node.arglist:
            self.walk(arg)

    def walk_Variable(self, node: ast.Variable):
        for arg in node.arglist:
            self.walk(arg)

    def walk_Constraints(self, node: ast.Constraints):
        self.walk(node.left)
        if node.right:
            self.walk(node.right)
