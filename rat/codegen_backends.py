from contextlib import contextmanager
from dataclasses import dataclass, field
from distutils.errors import CompileError
from . import ast
from tatsu.model import NodeWalker
from .variable_table import VariableTable
from .subscript_table import SubscriptTable
from typing import Set, List, Iterable


@dataclass
class IndentWriter:
    string: str = ""
    indent_spacing: int = 4
    indent_level: int = 0

    def writeline(self, string: str = ""):
        indent_string = self.indent_level * self.indent_spacing * " " if len(string) > 0 else ""
        self.string += f"{indent_string}{string}\n"

    @contextmanager
    def indent(self):
        try:
            self.indent_level += 1
            yield self
        finally:
            self.indent_level -= 1

    def __str__(self):
        return self.string


class BaseCodeGenerator(NodeWalker):
    variable_table: VariableTable
    subscript_table: SubscriptTable
    available_as_local: Set[str]
    left_side_of_sampling: ast.ModelBase
    extra_subscripts: List[str]

    def __init__(self, variable_table: VariableTable = None, subscript_table: SubscriptTable = None, available_as_local: Iterable[str] = None):
        self.available_as_local = set(available_as_local) if available_as_local else set()
        self.variable_table = variable_table
        self.subscript_table = subscript_table
        self.left_side_of_sampling = None
        self.extra_subscripts = []
        super().__init__()

    def walk_Statement(self, node: ast.Statement):
        if node.op == "~":
            self.left_side_of_sampling = node.left
            return f"({self.walk(node.right)})"
        else:
            raise CompileError(f"{node.op} operator not supported in BaseCodeGenerator", node)

    def walk_Logical(self, node: ast.Logical):
        return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Binary(self, node: ast.Binary):
        return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_IfElse(self, node: ast.IfElse):
        return f"({self.walk(node.left)} if {self.walk(node.predicate)} else {self.walk(node.right)})"

    def walk_FunctionCall(self, node: ast.FunctionCall):
        arglist = []

        if self.left_side_of_sampling:
            old_left_side_of_sampling = self.left_side_of_sampling
            self.left_side_of_sampling = None
            arglist += [self.walk(old_left_side_of_sampling)]
            self.left_side_of_sampling = old_left_side_of_sampling

        if node.arglist:
            arglist += [self.walk(arg) for arg in node.arglist]

        return f"{node.name}({','.join(arglist)})"

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.available_as_local:
            return node.name
        else:
            if node.arglist:
                record = self.subscript_table[node]
                self.extra_subscripts.append(record.name)
                index_lookup = f"[{record.name}]"
            else:
                index_lookup = ""

            return f"parameters['{node.name}']{index_lookup}"

    def walk_Literal(self, node: ast.Literal):
        return f"{node.value}"


class TraceCodeGenerator(NodeWalker):
    """
    As long as shunt is true, values of expressions aren't returned but
    instead shunted off into tuples
    """

    shunt: bool
    passed_by_value : Iterable[str]

    def __init__(self, passed_by_value : Iterable[str], shunt : bool = True):
        self.shunt = shunt
        self.passed_by_value = passed_by_value

    def walk_Statement(self, node: ast.Statement):
        if self.shunt:
            return f"({self.walk(node.left)}, {self.walk(node.right)})"
        else:
            return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Logical(self, node: ast.Logical):
        if self.shunt:
            return f"({self.walk(node.left)}, {self.walk(node.right)})"
        else:
            return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Binary(self, node: ast.Binary):
        if self.shunt:
            return f"({self.walk(node.left)}, {self.walk(node.right)})"
        else:
            return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_IfElse(self, node: ast.IfElse):
        old_shunt = self.shunt
        self.shunt = False
        predicate = self.walk(node.predicate)
        self.shunt = old_shunt

        return f"({self.walk(node.left)} if {predicate} else {self.walk(node.right)})"

    def walk_FunctionCall(self, node: ast.FunctionCall):
        if node.arglist:
            arglist = ",".join(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ""

        if self.shunt:
            return f"({arglist})"
        else:
            return f"{node.name}({arglist})"

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.passed_by_value:
            return f"values['{node.name}']"
        else:
            old_shunt = self.shunt
            self.shunt = False
            if node.arglist:
                arglist = ",".join(self.walk(arg) for arg in node.arglist)
            else:
                arglist = ""
            self.shunt = old_shunt

            return f"tracers['{node.name}']({arglist})"

    def walk_Literal(self, node: ast.Literal):
        return f"{node.value}"
