import jax

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, TypeVar, Generic, Union, Tuple, Optional

from tatsu.model import NodeWalker

from rat import ast
from rat import math
from rat.exceptions import CompileError, ExecuteException
from rat.variable_table import VariableTable, DynamicVariableRecord, RecurrentVariableRecord


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
    base_index: int
    recurrent_variable_name: Optional[str]
    left_side_of_sampling: Union[None, ast.ModelBase]
    collecting_traces: ContextStack[bool]
    values: Dict[ast.ModelBase, Union[int, float, str]]

    def __init__(
            self,
            variable_table: VariableTable,
            leaves: Dict[str, Union[int, float, str]],
            base_index: int,
            recurrent_variable_name: Optional[str] = None
    ):
        self.variable_table = variable_table
        self.leaves = leaves
        self.base_index = base_index
        self.recurrent_variable_name = recurrent_variable_name
        self.left_side_of_sampling = None
        self.collecting_traces = ContextStack(True)
        self.values = {}

    def walk_Statement(self, node: ast.Statement):
        # Let's just say an assignment evaluates to the right hand side?
        if node.op == "=":
            # Don't walk the left hand side cuz we're assuming the subscripts of the left hand side are not expressions
            #self.walk(node.left)
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
        with self.collecting_traces.push(False):
            predicate = self.walk(node.predicate)

        if predicate is None:
            msg = f"Unable to evaluate predicate at compile time. Ensure there are no parameter dependencies here"
            raise CompileError(msg, node)

        output = self.walk(node.left) if predicate else self.walk(node.right)

        if output is not None and self.collecting_traces.peek():
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
            try:
                python_function = getattr(math, node.name)
            except AttributeError:
                msg = f"Error calling unknown function {node.name}"
                raise CompileError(msg, node)
            output = python_function(*arglist)
        else:
            output = None

        if output is not None:
            self.values[node] = output
        return output

    def walk_Variable(self, node: ast.Variable):
        with self.collecting_traces.push(False):
            trace_by_reference = False
            if node.name in self.variable_table:
                node_variable = self.variable_table[node.name]
                if isinstance(node_variable, DynamicVariableRecord):
                    trace_by_reference = True

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
                    if trace_by_reference:
                        if len(arglist) > 0:
                            if node.name == self.recurrent_variable_name:
                                output = node_variable.get_index(*arglist) - self.base_index

                                if output >= 0:
                                    msg = (
                                        "All references to the primary variable on the right hand side of a recurrence"
                                        " relation must have negative offsets relative to the left hand side. Found"
                                        f" offset {output}"
                                    )
                                    raise CompileError(msg, node)
                            else:
                                output = node_variable.get_index(*arglist)
                        else:
                            output = None
                    else:
                        output = node_variable.get_value(*arglist)
                else:
                    output = None

        if output is not None and self.collecting_traces.peek():
            self.values[node] = output

        # This is tricky -- only return a value if we actually have one
        # For variables where we trace by reference we don't have anything
        # to return
        if not trace_by_reference:
            return output

    def walk_Literal(self, node: ast.Literal):
        output = node.value
        self.values[node] = output
        return output


class DomainDiscoveryExecutor(NodeWalker):
    """
    As long as the top value of avoid_returning_values is true, values of expressions aren't returned
    """

    variable_table: VariableTable
    leaves: Dict[str, Any]
    avoid_returning_values: ContextStack[bool]

    def __init__(self, variable_table: VariableTable, leaves: Dict[str, Any]):
        self.variable_table = variable_table
        self.leaves = leaves
        self.avoid_returning_values = ContextStack(True)

    def walk_Statement(self, node: ast.Statement):
        self.walk(node.left)
        self.walk(node.right)

    def walk_Binary(self, node: ast.Binary):
        left = self.walk(node.left)
        right = self.walk(node.right)
        if not self.avoid_returning_values.peek():
            return eval(f"left {node.op} right")

    def walk_IfElse(self, node: ast.IfElse):
        with self.avoid_returning_values.push(False):
            predicate = self.walk(node.predicate)

        return self.walk(node.left) if predicate else self.walk(node.right)

    def walk_FunctionCall(self, node: ast.FunctionCall):
        if node.arglist:
            arglist = tuple(self.walk(arg) for arg in node.arglist)
        else:
            arglist = ()

        if not self.avoid_returning_values.peek():
            return getattr(math, node.name)(*arglist)

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.leaves:
            return self.leaves[node.name]
        else:
            with self.avoid_returning_values.push(False):
                if node.arglist:
                    arglist = tuple(self.walk(arg) for arg in node.arglist)
                else:
                    arglist = ()

            variable = self.variable_table[node.name]
            if len(variable.subscripts) != len(arglist):
                msg = f"{node.name} should have {len(variable.subscripts)} subscript(s), found {len(arglist)}"
                raise CompileError(msg, node)

            if isinstance(variable, DynamicVariableRecord):
                variable.add(*arglist)
            else:
                return variable.get_value(*arglist)

    def walk_Literal(self, node: ast.Literal):
        return node.value


@dataclass
class CodeExecutor(NodeWalker):
    traced_values: Dict[ast.ModelBase, Any]
    parameters: Dict[str, Any]
    recurrent_variable_name: Optional[str] = None
    carry: Optional[Tuple] = None
    left_side_of_sampling: Union[None, ast.ModelBase] = field(default=None)

    def walk_Statement(self, node: ast.Statement):
        if node.op == "~":
            self.left_side_of_sampling = node.left
            return_value = self.walk(node.right)
            self.left_side_of_sampling = None
            return return_value
        else:
            raise ExecuteException(f"{node.op} operator not supported in {type(self)}", node)

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
            if node.name == self.recurrent_variable_name:
                trace = self.traced_values[node]
                return self.carry[trace + 1]
            elif node.name in self.parameters:
                trace = self.traced_values[node]
                return self.parameters[node.name][trace]
            else:
                return self.traced_values[node]
        else:
            return self.parameters[node.name]

    def walk_Literal(self, node: ast.Literal):
        return node.value
