import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Union

import numpy
import pandas

from . import ast
from .codegen_backends import OpportunisticExecutor, TraceExecutor, ContextStack
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


@dataclass
class CreateVariableWalker(RatWalker):
    variable_table: VariableTable
    data: Dict[str, pandas.DataFrame]
    in_control_flow: ContextStack[bool] = field(default_factory=lambda: ContextStack(False))

    def walk_Statement(self, node: ast.Statement):
        if node.op != "=":
            self.walk(node.left)
        self.walk(node.right)

    def walk_IfElse(self, node: ast.IfElse):
        with self.in_control_flow.push(True):
            self.walk(node.predicate)

        self.walk(node.left)
        self.walk(node.right)

    def walk_Variable(self, node: ast.Variable):
        unallocated = node.name not in self.variable_table
        subscript_names = _get_subscript_names(node)

        if node.arglist:
            argument_count = len(node.arglist)
        else:
            argument_count = 0

        if unallocated:
            if self.in_control_flow.peek():
                record = ConstantVariableRecord(node.name, argument_count)
                try:
                    record.bind(subscript_names, self.data)
                except KeyError:
                    if argument_count > 0:
                        msg = f"{node.name} is in control flow and must bind to input data but that failed"
                        raise CompileError(msg, node)
                    else:
                        # It's okay if an argument_count == 0 something doesn't bind -- it (hopefully) will be provided by
                        #  the primary variable instead
                        return
            else:
                record = ConstantVariableRecord(node.name, argument_count)
                try:
                    record.bind(subscript_names, self.data)
                except KeyError:
                    record = SampledVariableRecord(node.name, argument_count)

            self.variable_table[node.name] = record
        else:
            record = self.variable_table[node.name]
            if isinstance(record, ConstantVariableRecord):
                try:
                    record.bind(subscript_names, self.data)
                except KeyError as e:
                    if argument_count > 0:
                        raise CompileError(str(e), node)
                    else:
                        # I think it's okay if an argument_count == 0 something doesn't bind here.
                        return

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


@dataclass
class SubscriptTableWalker(RatWalker):
    variable_table: VariableTable
    trace_table: TraceTable = field(default_factory=TraceTable)
    trace_by_reference: Set[ast.Variable] = field(default_factory=set)

    def walk_Statement(self, node: ast.Statement):
        primary_node = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_node.name]

        # Identify variables we won't know the value of yet -- don't try to
        # trace the values of those but trace any indexing into them
        self.trace_by_reference = set()

        self.walk(node.left)
        self.walk(node.right)

        traces = defaultdict(lambda: [])
        for row in primary_variable.itertuples():
            executor = OpportunisticExecutor(self.variable_table, row._asdict(), self.trace_by_reference)
            executor.walk(node)

            for traced_node, value in executor.values.items():
                traces[traced_node].append(value)

        for traced_node, values in traces.items():
            self.trace_table.insert(traced_node, numpy.array(values))

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.variable_table:
            node_variable = self.variable_table[node.name]
            if isinstance(node_variable, DynamicVariableRecord):
                self.trace_by_reference.add(node)


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


# 5. Check that the predicate of IfElse statement contains no parameters
@dataclass
class IfElsePredicateCheckWalker(RatWalker):
    variable_table: VariableTable
    inside_predicate: bool = False

    def walk_IfElse(self, node: ast.IfElse):
        old_inside_predicate = self.inside_predicate
        self.inside_predicate = True
        self.walk(node.predicate)
        self.inside_predicate = old_inside_predicate

        self.walk(node.left)
        self.walk(node.right)

    def walk_Variable(self, node: ast.Variable):
        if self.inside_predicate and isinstance(self.variable_table[node.name], DynamicVariableRecord):
            msg = f"Non-data variables cannot appear in if-else conditions"
            raise CompileError(msg, node)


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

        walker = IfElsePredicateCheckWalker(self.variable_table)
        walker.walk(program)
