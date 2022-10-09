import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Union

import numpy
import pandas

from . import ast
from .codegen_backends import SubscriptTableWalker, TraceExecutor, ContextStack, get_primary_ast_variable
from .exceptions import CompileError
from .trace_table import TraceTable
from .variable_table import VariableTable, AssignedVariableRecord, SampledVariableRecord, ConstantVariableRecord, DynamicVariableRecord
from .walker import RatWalker, NodeWalker


class NameWalker(RatWalker):
    @staticmethod
    def combine(names: List[str]):
        return functools.reduce(lambda x, y: x + y, filter(None, names))

    def walk_Binary(self, node: ast.Binary):
        return self.combine([self.walk(node.left), self.walk(node.right)])

    def walk_FunctionCall(self, node: ast.FunctionCall):
        return self.combine([self.walk(arg) for arg in node.arglist])

    def walk_Variable(self, node: ast.Variable):
        return {node.name}


def _get_subscript_names(node: ast.Variable) -> Union[None, List[str]]:
    """
    Examine every subscript argument to see if it has only one named variable

    If this is true for all arguments, return a list with these names
    """
    if node.arglist is None:
        return None

    names = []
    walker = NameWalker()
    for arg in node.arglist:
        arg_names = list(walker.walk(arg))

        if len(arg_names) == 1:
            names.append(arg_names[0])
        else:
            # It wasn't possible to get a unique name for each subscript
            return None
    return names


# 1. If the variable on the left appears also on the right, mark this
#   statement to be code-generated with a scan and also make sure that
#   the right hand side is shifted appropriately
@dataclass
class CreateAssignedVariablesWalker(RatWalker):
    variable_table: VariableTable

    def walk_Statement(self, node: ast.Statement):
        if node.op == "=":
            assigned_name = self.walk(node.left)
            if not assigned_name:
                msg = "The left hand side of an assignment must be a variable"
                raise CompileError(msg, node.left)
            return assigned_name

    def walk_Variable(self, node: ast.Variable):
        self.variable_table[node.name] = AssignedVariableRecord(node.name, len(node.arglist) if node.arglist else 0)
        return True


# Check that subscripts of primary variables don't include expressions
@dataclass
class CheckPrimaryVariableSubscriptWalker(NodeWalker):
    msg: str = "Primary variables cannot have expressions in their subscripts"

    def walk_Statement(self, node: ast.Statement):
        if node.op == "~":
            if not self.walk(node.right):
                raise CompileError(self.msg, node.right)

    def walk_Variable(self, node: ast.Variable):
        if node.arglist:
            for arg in node.arglist:
                if not self.walk(arg):
                    raise CompileError(self.msg, node)
        return True


@dataclass
class CreateVariableWalker(RatWalker):
    variable_table: VariableTable
    data: Dict[str, pandas.DataFrame]
    in_control_flow: ContextStack[bool] = field(default_factory=lambda: ContextStack(False))
    primary_node_and_subscript_names: Set[str] = field(default_factory=set)

    def walk_Statement(self, node: ast.Statement):
        primary_node = get_primary_ast_variable(node)
        self.primary_node_and_subscript_names = (
            {primary_node.name} | {node.name for node in primary_node.arglist} if primary_node.arglist else {}
        )

        if node.op != "=":
            self.walk(node.left)
        self.walk(node.right)

    def walk_IfElse(self, node: ast.IfElse):
        with self.in_control_flow.push(True):
            self.walk(node.predicate)

        self.walk(node.left)
        self.walk(node.right)

    def walk_Variable(self, node: ast.Variable):
        unknown = node.name not in self.variable_table
        subscript_names = _get_subscript_names(node)

        if node.arglist:
            argument_count = len(node.arglist)
        else:
            argument_count = 0

        if unknown:
            if self.in_control_flow.peek():
                if argument_count == 0:
                    if node.name not in self.primary_node_and_subscript_names:
                        msg = f"{node.name} is in control flow and must be a primary variable subscript, but it is not"
                        raise CompileError(msg, node)
                    else:
                        return

                record = ConstantVariableRecord(node.name, argument_count)

                try:
                    record.bind(subscript_names, self.data)
                except KeyError as e:
                    msg = f"{node.name} is in control flow and must bind to input data but that failed"
                    raise CompileError(msg, node) from e
                except Exception as e:
                    msg = f"Irrecoverable error binding {node.name} in control flow to data"
                    raise CompileError(msg, node) from e
            else:
                record = ConstantVariableRecord(node.name, argument_count)
                try:
                    record.bind(subscript_names, self.data)
                except KeyError:
                    record = SampledVariableRecord(node.name, argument_count)
                except Exception as e:
                    msg = f"Irrecoverable error binding {node.name} to data"
                    raise CompileError(msg, node) from e

            self.variable_table[node.name] = record
        else:
            record = self.variable_table[node.name]

            if isinstance(record, ConstantVariableRecord):
                if argument_count == 0:
                    if node.name not in self.primary_node_and_subscript_names:
                        msg = (
                            f"{node.name} is a constant variable and must be a primary variable subscript if it appears with no subscripts"
                        )
                        raise CompileError(msg, node)
                    else:
                        return

                try:
                    record.bind(subscript_names, self.data)
                except KeyError as e:
                    msg = f"Failed to bind data to constant variable {node.name}"
                    raise CompileError(msg, node) from e
                except Exception as e:
                    msg = f"Irrecoverable error binding constant variable {node.name} in control flow to data"
                    raise CompileError(msg, node) from e

        if node.arglist:
            with self.in_control_flow.push(True):
                for arg in node.arglist:
                    self.walk(arg)


@dataclass
class RenameSubscriptWalker(RatWalker):
    variable_table: VariableTable

    def walk_Statement(self, node: ast.Statement):
        # Find the primary variable
        primary_ast_variable = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_ast_variable.name]
        primary_subscript_names = _get_subscript_names(primary_ast_variable)

        if primary_subscript_names:
            try:
                primary_variable.rename(primary_subscript_names)
            except AttributeError:
                msg = (
                    f"Attempting to rename subscripts to {primary_subscript_names} but they have already"
                    " been renamed to {primary_variable.subscripts}"
                )
                raise CompileError(msg, primary_ast_variable)


# Do a sweep to infer subscript names for subscripts not renamed
@dataclass
class InferSubscriptNameWalker(RatWalker):
    variable_table: VariableTable
    primary_name: str = None

    def walk_Statement(self, node: ast.Statement):
        primary_variable = get_primary_ast_variable(node)
        self.primary_name = primary_variable.name

    def walk_Variable(self, node: ast.Variable):
        if node.name != self.primary_name:
            if node.arglist:
                variable = self.variable_table[node.name]
                subscript_names = _get_subscript_names(node)

                if isinstance(variable, DynamicVariableRecord) and not variable.renamed and subscript_names is not None:
                    try:
                        variable.suggest_names(subscript_names)
                    except AttributeError:
                        msg = (
                            f"Attempting to reference subscript of {node.name} as {subscript_names}, but"
                            " they have already been referenced as {variable.subscripts}. The subscripts"
                            " must be renamed"
                        )
                        raise CompileError(msg, node)

        if node.arglist:
            for arg in node.arglist:
                self.walk(arg)


class DomainDiscoveryWalker(RatWalker):
    variable_table: VariableTable

    def __init__(self, variable_table: VariableTable):
        super().__init__()

        self.variable_table = variable_table

        for name in self.variable_table:
            variable = self.variable_table[name]
            if isinstance(variable, DynamicVariableRecord):
                variable.swap_and_clear_write_buffer()

    def walk_Statement(self, node: ast.Statement):
        # Find the primary variable
        primary_ast_variable = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_ast_variable.name]

        for row in primary_variable.itertuples():
            executor = TraceExecutor(self.variable_table, row._asdict())
            executor.walk(node)


# Apply constraints to parameters
@dataclass
class SingleConstraintCheckWalker(RatWalker):
    variable_table: VariableTable
    primary_name: str = None
    found_constraints_for: Set[str] = field(default_factory=set)

    def walk_Statement(self, node: ast.Statement):
        self.primary_name = get_primary_ast_variable(node).name
        self.walk(node.left)
        self.walk(node.right)

    def walk_Variable(self, node: ast.Variable):
        if node.constraints is not None:
            if self.primary_name != node.name:
                msg = f"Attempting to set constraints on {node.name} which is not the primary variable" f" ({self.primary_name})"
                raise CompileError(msg, node)
            else:
                if node.name in self.found_constraints_for:
                    msg = f"Attempting to set constraints on {node.name} but they have previously been set"
                    raise CompileError(msg, node)
                else:
                    self.found_constraints_for.add(node.name)


# 2. Check that the right hand side of a sampling statement is a
#   function call
@dataclass
class CheckSamplingFunctionWalker(NodeWalker):
    variable_table: VariableTable

    def walk_Statement(self, node: ast.Statement):
        if node.op == "~":
            if not self.walk(node.right):
                msg = "The right hand side of a sampling statement must be a function"
                raise CompileError(msg, node.right)

    def walk_FunctionCall(self, node: ast.FunctionCall):
        return True


# 5. Check that control flow contains no parameters
@dataclass
class ControlFlowParameterCheckWalker(RatWalker):
    variable_table: VariableTable
    in_control_flow: ContextStack[bool] = field(default_factory=lambda: ContextStack(False))

    def walk_IfElse(self, node: ast.IfElse):
        with self.in_control_flow.push(True):
            self.walk(node.predicate)

        self.walk(node.left)
        self.walk(node.right)

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.variable_table:
            if self.in_control_flow.peek() and isinstance(self.variable_table[node.name], DynamicVariableRecord):
                msg = f"Non-data variables cannot appear in if-else conditions"
                raise CompileError(msg, node)

            if node.arglist:
                with self.in_control_flow.push(True):
                    for arg in node.arglist:
                        self.walk(arg)


# 4. Parameters cannot be assigned after they are referenced
@dataclass
class CheckTransformedParameterOrder(RatWalker):
    referenced: Set[str] = field(default_factory=set)

    def walk_Statement(self, node: ast.Statement):
        if node.op == "=":
            node_name = self.walk(node.left)
            if node_name is not None:
                if node_name in self.referenced:
                    msg = f"Variable {node_name} cannot be assigned after it is used"
                    raise CompileError(msg, node.left)

                self.referenced.add(node_name)

    def walk_Variable(self, node: ast.Variable):
        return node.name


class RatCompiler:
    variable_table: VariableTable
    trace_table: TraceTable

    def __init__(self, data: Dict[str, pandas.DataFrame], program: ast.Program, max_trace_iterations: int):
        self.variable_table = VariableTable()

        walker = CheckPrimaryVariableSubscriptWalker()
        walker.walk(program)

        walker = CreateAssignedVariablesWalker(self.variable_table)
        walker.walk(program)

        walker = CreateVariableWalker(self.variable_table, data)
        walker.walk(program)

        # Do a sweep to rename the primary variables as necessary
        walker = RenameSubscriptWalker(self.variable_table)
        walker.walk(program)

        walker = InferSubscriptNameWalker(self.variable_table)
        walker.walk(program)

        # bind_data_to_functions_walker = BindDataToFunctionsWalker(self.variable_table, data)
        # bind_data_to_functions_walker.walk(program)

        # Trace the program to determine parameter domains and check data domains
        for _ in range(max_trace_iterations):
            walker = DomainDiscoveryWalker(self.variable_table)
            walker.walk(program)

            any_domain_changed = False
            for variable in self.variable_table.variables():
                if isinstance(variable, DynamicVariableRecord):
                    any_domain_changed |= variable.buffers_are_different()

            if not any_domain_changed:
                break
        else:
            raise CompileError(f"Unable to resolve subscripts after {max_trace_iterations} trace iterations")

        walker = SingleConstraintCheckWalker(self.variable_table)
        walker.walk(program)

        walker = SubscriptTableWalker(self.variable_table)
        walker.walk(program)
        self.trace_table = walker.trace_table

        walker = CheckSamplingFunctionWalker(self.variable_table)
        walker.walk(program)

        walker = CheckTransformedParameterOrder()
        walker.walk(program)

        walker = ControlFlowParameterCheckWalker(self.variable_table)
        walker.walk(program)
