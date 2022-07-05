from dataclasses import dataclass
from . import ast2
from tatsu.model import NodeWalker


@dataclass
class BaseCodeGenerator(NodeWalker):
    def walk_Statement(self, node: ast2.Statement):
        return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Logical(self, node: ast2.Logical):
        return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Binary(self, node: ast2.Binary):
        return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_IfElse(self, node: ast2.IfElse):
        return f"({self.walk(node.left)} if {self.walk(node.predicate)} else {self.walk(node.right)})"

    def walk_FunctionCall(self, node: ast2.FunctionCall):
        if node.arglist:
            arglist = ",".join(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ""

        return f"{node.name}({arglist})"

    def walk_Variable(self, node: ast2.Variable):
        if node.arglist:
            arglist = ",".join(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ""

        return f"{node.name}({arglist})"

    def walk_Literal(self, node: ast2.Literal):
        return f"{node.value}"


@dataclass
class DiscoverVariablesCodeGenerator(NodeWalker):
    """
    As long as shunt is true, values of expressions aren't returned but
    instead shunted off into tuples
    """

    shunt = True

    def walk_Statement(self, node: ast2.Statement):
        if self.shunt:
            return f"({self.walk(node.left)}, {self.walk(node.right)})"
        else:
            return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Logical(self, node: ast2.Logical):
        if self.shunt:
            return f"({self.walk(node.left)}, {self.walk(node.right)})"
        else:
            return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Binary(self, node: ast2.Binary):
        if self.shunt:
            return f"({self.walk(node.left)}, {self.walk(node.right)})"
        else:
            return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_IfElse(self, node: ast2.IfElse):
        old_shunt = self.shunt
        self.shunt = False
        predicate = self.walk(node.predicate)
        self.shunt = old_shunt

        return f"({self.walk(node.left)} if {predicate} else {self.walk(node.right)})"

    def walk_FunctionCall(self, node: ast2.FunctionCall):
        if node.arglist:
            arglist = ",".join(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ""

        if self.shunt:
            return f"({arglist})"
        else:
            return f"{node.name}({arglist})"

    def walk_Variable(self, node: ast2.Variable):
        if node.arglist:
            arglist = ",".join(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ""

        return f"{node.name}({arglist})"

    def walk_Literal(self, node: ast2.Literal):
        return f"{node.value}"
