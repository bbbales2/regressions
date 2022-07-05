from . import ast2
from tatsu.model import NodeWalker
from tatsu.model import ModelBuilderSemantics


class RatWalker(NodeWalker):
    def walk_Program(self, node: ast2.Program):
        for statement in node.statements:
            self.walk(statement)

    def walk_Statement(self, node: ast2.Statement):
        self.walk(node.left)
        self.walk(node.right)

    def walk_Logical(self, node: ast2.Logical):
        self.walk(node.left)
        self.walk(node.right)

    def walk_Binary(self, node: ast2.Binary):
        self.walk(node.left)
        self.walk(node.right)

    def walk_IfElse(self, node: ast2.IfElse):
        self.walk(node.predicate)
        self.walk(node.left)
        self.walk(node.right)

    def walk_FunctionCall(self, node: ast2.FunctionCall):
        for arg in node.arglist:
            self.walk(arg)

    def walk_Variable(self, node: ast2.Variable):
        for arg in node.arglist:
            self.walk(arg)

    def walk_Constraints(self, node: ast2.Constraints):
        self.walk(node.left)
        if node.right:
            self.walk(node.right)
