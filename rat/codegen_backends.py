import jax
import numpy
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Set, List, Dict, Any, TypeVar, Generic, Union

from tatsu.model import NodeWalker

from . import ast
from . import math
from .exceptions import CompileError, ExecuteException
from .variable_table import VariableTable, DynamicVariableRecord
from .trace_table import TraceTable
from .walker import RatWalker


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


def get_primary_ast_variable(statement: ast.Statement) -> ast.Variable:
    """
    Compute the primary variable reference in a line of code with the rules:
    1. There can only be one primary variable reference (priming two references to the same variable is still an error)
    2. If a variable is marked as primary, then it is the primary variable.
    3. If there is no marked primary variable, then all variables with dataframes are treated as prime.
    4. If there are no variables with dataframes, the leftmost one is the primary one
    5. It is an error if no primary variable can be identified
    """

    @dataclass
    class PrimaryWalker(RatWalker):
        marked: ast.Variable = None
        candidates: List[ast.Variable] = field(default_factory=list)

        def walk_Variable(self, node: ast.Variable):
            if node.prime:
                if self.marked is None:
                    self.marked = node
                else:
                    msg = f"Found two marked primary variables {self.marked.name} and {node.name}. There should only" "be one"
                    raise CompileError(msg, node)
            else:
                self.candidates.append(node)

    walker = PrimaryWalker()
    walker.walk(statement)
    marked = walker.marked
    candidates = walker.candidates

    if marked is not None:
        return marked

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        if len(set(candidate.name for candidate in candidates)) == 1:
            msg = (
                f"No marked primary variable but found multiple references to {candidates[0].name}. One reference"
                " should be marked manually"
            )
            raise CompileError(msg, candidates[0])
        else:
            msg = (
                f"No marked primary variable and at least {candidates[0].name} and {candidates[1].name} are"
                " candidates. A primary variable should be marked manually"
            )
            raise CompileError(msg, candidates[0])

    if len(candidates) == 0:
        msg = f"No primary variable found on line (this means there are no candidate variables)"
        raise CompileError(msg, statement)


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
                    output = self.variable_table[node.name].get_index(arglist)
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
                msg = f"{node.name} should have {len(variable.subscripts)} subscripts, found {len(arglist)}"
                raise CompileError(msg, node)
            return variable(*arglist)

    def walk_Literal(self, node: ast.Literal):
        return node.value


class CodeExecutor(NodeWalker):
    traced_values: Dict[ast.Variable, Any]
    parameters: Dict[str, Any]
    left_side_of_sampling: Union[None, ast.ModelBase]

    def __init__(self, traced_values: Dict[ast.Variable, Any] = None, parameters: Dict[str, Any] = None):
        self.traced_values = traced_values
        self.parameters = parameters
        self.left_side_of_sampling = None
        super().__init__()

    def walk_Statement(self, node: ast.Statement):
        if node.op == "~":
            self.left_side_of_sampling = node.left
            return_value = self.walk(node.right)
            self.left_side_of_sampling = None
            return return_value
        else:
            raise ExecuteException(f"{node.op} operator not supported in BaseCodeGenerator", node)

    def walk_Binary(self, node: ast.Binary):
        left = self.walk(node.left)
        right = self.walk(node.right)
        return eval(f"left {node.op} right")

    def walk_IfElse(self, node: ast.IfElse):
        predicate = self.traced_values[node.predicate]
        return jax.lax.cond(predicate, lambda: self.walk(node.left), lambda: self.walk(node.right))

    def walk_FunctionCall(self, node: ast.FunctionCall):
        argument_list = []

        if self.left_side_of_sampling:
            argument_list += [self.walk(self.left_side_of_sampling)]

        if node.arglist:
            argument_list += [self.walk(arg) for arg in node.arglist]

        return getattr(math, node.name)(*argument_list)

    def walk_Variable(self, node: ast.Variable):
        if node in self.traced_values:
            if node.name in self.parameters:
                trace = self.traced_values[node]
                return self.parameters[node.name][trace]
            else:
                return self.traced_values[node]
        else:
            return self.parameters[node.name]

    def walk_Literal(self, node: ast.Literal):
        return node.value


@dataclass
class DynamicVariableRecordFinderWalker(RatWalker):
    variable_table: VariableTable
    dynamic_variable_nodes: Set[ast.Node] = field(default_factory=set)

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.variable_table:
            node_variable = self.variable_table[node.name]
            if isinstance(node_variable, DynamicVariableRecord):
                self.dynamic_variable_nodes.add(node)


@dataclass
class SubscriptTableWalker(RatWalker):
    variable_table: VariableTable
    trace_table: TraceTable = field(default_factory=TraceTable)

    def process_node(self, node: ast.ModelBase):
        primary_node = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_node.name]

        # Identify variables we won't know the value of yet -- don't try to
        # trace the values of those but trace any indexing into them
        self.trace_by_reference = set()

        walker = DynamicVariableRecordFinderWalker(self.variable_table)
        walker.walk(node)
        trace_by_reference = walker.dynamic_variable_nodes

        traces = defaultdict(lambda: [])
        for row in primary_variable.itertuples():
            executor = OpportunisticExecutor(self.variable_table, row._asdict(), trace_by_reference)
            executor.walk(node)

            for traced_node, value in executor.values.items():
                traces[traced_node].append(value)

        for traced_node, values in traces.items():
            self.trace_table.insert(traced_node, numpy.array(values))


    def walk_Statement(self, node: ast.Statement):
        self.process_node(node)
