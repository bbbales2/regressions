from contextlib import contextmanager
from dataclasses import dataclass
from typing import Set, List, Dict, Any, TypeVar, Generic, Union

from tatsu.model import NodeWalker

from . import ast
from . import math
from .exceptions import CompileError
from .variable_table import VariableTable


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


T = TypeVar("T")


@dataclass
class ContextStack(Generic[T]):
    stack: List[T]

    def __init__(self, first: T):
        self.stack = [first]

    def peek(self) -> T:
        return self.stack[-1]

    @contextmanager
    def push(self, value: T):
        try:
            self.stack.append(value)
            yield self
        finally:
            self.stack.pop()


class OpportunisticExecutor(NodeWalker):
    """
    Execute as much of the code as possible
    """

    variable_table: VariableTable
    leaves: Dict[str, Any]
    trace_by_reference: Set[ast.ModelBase]
    left_side_of_sampling: Union[None, ast.ModelBase]
    values: Dict[ast.ModelBase, Union[int, float, str]]

    def __init__(self, variable_table: VariableTable, leaves: Dict[str, Union[int, float, str]], trace_by_reference: Set[ast.ModelBase]):
        self.variable_table = variable_table
        self.leaves = leaves
        self.trace_by_reference = trace_by_reference
        self.left_side_of_sampling = None
        self.values = {}

    def walk_Statement(self, node: ast.Statement):
        # Let's just say an assignment evaluates to the right hand side?
        if node.op == "=":
            self.walk(node.left)
            output = self.walk(node.right)
        else:
            self.left_side_of_sampling = node.left
            output = self.walk(node.right)

        if output is not None:
            self.values[node] = output
        return output

    def walk_Binary(self, node: ast.Binary):
        left = self.walk(node.left)
        right = self.walk(node.right)
        if left is not None and right is not None:
            output = eval(f"left {node.op} right")
        else:
            output = None

        if output is not None:
            self.values[node] = output
        return output

    def walk_IfElse(self, node: ast.IfElse):
        predicate = self.walk(node.predicate)

        if predicate is None:
            msg = f"Unable to evaluate predicate at compile time. Ensure there are no parameter dependencies here"
            raise CompileError(msg, node)

        output = self.walk(node.left) if predicate else self.walk(node.right)

        if output is not None:
            self.values[node] = output
        return output

    def walk_FunctionCall(self, node: ast.FunctionCall):
        if self.left_side_of_sampling:
            initial_arglist = (self.walk(self.left_side_of_sampling),)
            self.left_side_of_sampling = None
        else:
            initial_arglist = ()

        if node.arglist:
            arglist = initial_arglist + tuple(self.walk(arg) for arg in node.arglist)
        else:
            arglist = initial_arglist

        if all(arg is not None for arg in arglist):
            output = getattr(math, node.name)(*arglist)
        else:
            output = None

        if output is not None:
            self.values[node] = output
        return output

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.leaves:
            output = self.leaves[node.name]
        else:
            if node.constraints:
                constraint_msg = f"Unable to evaluate constraint at compile time. Ensure there are no parameter dependencies here"
                if node.constraints.left:
                    left_constraint_value = self.walk(node.constraints.left.value)
                    if left_constraint_value is None:
                        raise CompileError(constraint_msg, node.constraints.left)

                if node.constraints.right:
                    right_constraint_value = self.walk(node.constraints.right.value)
                    if right_constraint_value is None:
                        raise CompileError(constraint_msg, node.constraints.right)

            if node.arglist:
                arglist = tuple(self.walk(arg) for arg in node.arglist)
            else:
                arglist = ()

            if all(arg is not None for arg in arglist):
                if node in self.trace_by_reference:
                    if len(arglist) > 0:
                        output = self.variable_table[node.name].get_index(arglist)
                    else:
                        output = None
                else:
                    output = self.variable_table[node.name].get_value(arglist)
            else:
                output = None

        if output is not None:
            self.values[node] = output

        # This is tricky -- only return a value if we actually have one
        # For variables where we trace by reference we don't have anything
        # to return
        if node not in self.trace_by_reference:
            return output

    def walk_Literal(self, node: ast.Literal):
        output = node.value
        self.values[node] = output
        return output


class TraceExecutor(NodeWalker):
    """
    As long as shunt is true, values of expressions aren't returned but
    instead shunted off into tuples
    """

    variable_table: VariableTable
    leaves: Dict[str, Any]
    shunt: ContextStack[bool]

    def __init__(self, variable_table: VariableTable, leaves: Dict[str, Any]):
        self.variable_table = variable_table
        self.leaves = leaves
        self.shunt = ContextStack(True)

    def walk_Statement(self, node: ast.Statement):
        self.walk(node.left)
        self.walk(node.right)

    def walk_Binary(self, node: ast.Binary):
        left = self.walk(node.left)
        right = self.walk(node.right)
        if not self.shunt.peek():
            return eval(f"left {node.op} right")

    def walk_IfElse(self, node: ast.IfElse):
        with self.shunt.push(False):
            predicate = self.walk(node.predicate)

        return self.walk(node.left) if predicate else self.walk(node.right)

    def walk_FunctionCall(self, node: ast.FunctionCall):
        if node.arglist:
            arglist = tuple(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ()

        if not self.shunt.peek():
            return getattr(math, node.name)(*arglist)

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.leaves:
            return self.leaves[node.name]
        else:
            with self.shunt.push(False):
                if node.arglist:
                    arglist = tuple(self.walk(arg) for arg in node.arglist)
                else:
                    arglist = ()

            variable = self.variable_table[node.name]
            if len(variable.subscripts) != len(arglist):
                msg = f"{node.name} should have {len(variable.subscripts)} subscript(s), found {len(arglist)}"
                raise CompileError(msg, node)
            return variable(*arglist)

    def walk_Literal(self, node: ast.Literal):
        return node.value
