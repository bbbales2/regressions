from contextlib import contextmanager
from dataclasses import dataclass
from distutils.errors import CompileError
from typing import Set, List, Iterable, Dict, Any, TypeVar, Generic, Union

from tatsu.model import NodeWalker

from . import ast
from . import math
from .subscript_table import SubscriptTable
from .variable_table import VariableTable, VariableType, Tracer


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


class BaseCodeGenerator(NodeWalker):
    variable_table: VariableTable
    subscript_table: SubscriptTable
    available_as_local: Set[str]
    left_side_of_sampling: ast.ModelBase
    used_arguments: List[str]
    extra_subscripts: List[str]

    def __init__(
            self, variable_table: VariableTable = None, subscript_table: SubscriptTable = None,
            available_as_local: Iterable[str] = None
    ):
        self.available_as_local = set(available_as_local) if available_as_local else set()
        self.variable_table = variable_table
        self.subscript_table = subscript_table
        self.left_side_of_sampling = None
        self.extra_subscripts = []
        self.used_arguments = []
        super().__init__()

    def walk_Statement(self, node: ast.Statement):
        if node.op == "~":
            self.left_side_of_sampling = node.left
            return f"({self.walk(node.right)})"
        else:
            raise CompileError(f"{node.op} operator not supported in BaseCodeGenerator", node)

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
            self.used_arguments.append(node.name)
            return node.name
        else:
            variable = self.variable_table[node.name]

            if variable.variable_type == VariableType.DATA:
                try:
                    record = self.subscript_table[node]
                    self.extra_subscripts.append(record.name)
                except KeyError:
                    raise Exception(
                        f"Internal compiler error: there should be a trace for every data variable in the ast")

                return record.name
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


class OpportunisticExecutor(NodeWalker):
    """
    Execute as much of the code as possible
    """
    tracers: Dict[str, Tracer]
    leaves: Dict[str, Any]
    trace_by_reference: Set[ast.ModelBase]
    left_side_of_sampling: ast.ModelBase
    values: Dict[ast.ModelBase, Union[int, float, str]]

    def __init__(self, tracers: Dict[str, Tracer], leaves: Dict[str, Union[int, float, str]],
                 trace_by_reference: Set[ast.ModelBase]):
        self.tracers = tracers
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
                    output = self.tracers[node.name].lookup(arglist)
                else:
                    output = self.tracers[node.name](*arglist)
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

    tracers: Dict[str, Tracer]
    leaves: Dict[str, Any]
    shunt: ContextStack[bool]
    trace_by_reference: Set[ast.Variable]
    first_level_values: Dict[ast.Variable, Any]

    def __init__(self, tracers: Dict[str, Tracer], leaves: Dict[str, Any],
                 trace_by_reference: Set[ast.Variable] = None):
        self.tracers = tracers
        self.leaves = leaves
        self.shunt = ContextStack(True)
        self.trace_by_reference = trace_by_reference if trace_by_reference else set()
        self.first_level_values = {}

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

        if self.shunt.peek():
            self.first_level_values[node] = predicate

        return self.walk(node.left) if predicate else self.walk(node.right)

    def walk_FunctionCall(self, node: ast.FunctionCall):
        if node.arglist:
            arglist = tuple(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ()

        if not self.shunt.peek():
            return eval(node.name)(*arglist)

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.leaves:
            return_value = self.leaves[node.name]
        else:
            with self.shunt.push(False):
                if node.arglist:
                    arglist = tuple(self.walk(arg) for arg in node.arglist)
                else:
                    arglist = ()

            if node in self.trace_by_reference:
                return_value = self.tracers[node.name].lookup(arglist)
            else:
                return_value = self.tracers[node.name](*arglist)

        if self.shunt.peek():
            self.first_level_values[node] = return_value

        return return_value

    def walk_Literal(self, node: ast.Literal):
        return node.value
