import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Union

import numpy
import pandas

from . import ast
from .codegen_backends import OpportunisticExecutor, TraceExecutor
from .exceptions import CompileError
from .subscript_table import SubscriptTable
from .variable_table import Tracer, VariableTable, VariableRecord, VariableType
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


# Add entries to the variable table for all ast variables
@dataclass
class CreateVariableWalker(RatWalker):
    variable_table: VariableTable
    in_control_flow: bool = False
    left_hand_of_assignment: bool = False

    def walk_Statement(self, node: ast.Statement):
        if node.op == "=":
            old_left_hand_of_assignment = self.left_hand_of_assignment
            self.left_hand_of_assignment = True
            self.walk(node.left)
            self.left_hand_of_assignment = old_left_hand_of_assignment
        else:
            self.walk(node.left)
        self.walk(node.right)

    def walk_IfElse(self, node: ast.IfElse):
        self.in_control_flow = True
        self.walk(node.predicate)
        self.in_control_flow = False

        self.walk(node.left)
        self.walk(node.right)

    def walk_Variable(self, node: ast.Variable):
        if node.arglist:
            argument_count = len(node.arglist)
        else:
            argument_count = 0

        if self.in_control_flow:
            # Variable nodes without subscripts in control flow must come from primary variable
            if node.arglist:
                self.variable_table.insert(variable_name=node.name, argument_count=argument_count, variable_type=VariableType.DATA)
        else:
            # Overwrite table entry for assigned parameters
            # (so they don't get turned back into regular parameters)
            if self.left_hand_of_assignment:
                self.variable_table.insert(
                    variable_name=node.name, argument_count=argument_count, variable_type=VariableType.ASSIGNED_PARAM
                )
            else:
                if node.name not in self.variable_table:
                    self.variable_table.insert(variable_name=node.name, argument_count=argument_count, variable_type=VariableType.PARAM)

        if node.arglist:
            self.in_control_flow = True
            for arg in node.arglist:
                self.walk(arg)
            self.in_control_flow = False


@dataclass
class RenameWalker(RatWalker):
    variable_table: VariableTable

    def walk_Statement(self, node: ast.Statement):
        # Find the primary variable
        primary_ast_variable = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_ast_variable.name]
        primary_subscript_names = _get_subscript_names(primary_ast_variable)

        if primary_subscript_names:
            # TODO: Make property?
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

                if variable.variable_type != VariableType.DATA and not variable.renamed and subscript_names is not None:
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


# Greedily try to bind variables to data
@dataclass
class BindDataToFunctionsWalker(RatWalker):
    variable_table: VariableTable
    data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]

    def walk_Variable(self, node: ast.Variable):
        # Do not bind variables without an argument list
        if node.arglist:
            subscript_names = _get_subscript_names(node)

            variable = self.variable_table[node.name]
            variable.bind(subscript_names, self.data)

            for arg in node.arglist:
                self.walk(arg)


@dataclass
class DomainDiscoveryWalker(RatWalker):
    variable_table: VariableTable

    def __init__(self, variable_table: VariableTable):
        self.variable_table = variable_table

        # Reset all the parameter tracers
        self.tracers = {}
        for name in self.variable_table:
            variable = self.variable_table[name]
            if not variable.tracer:
                variable.tracer = Tracer()
            self.tracers[name] = variable.tracer.copy()

    def walk_Statement(self, node: ast.Statement):
        # Find the primary variable
        primary_ast_variable = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_ast_variable.name]

        for row in primary_variable.itertuples():
            executor = TraceExecutor(self.tracers, row._asdict())
            executor.walk(node)


@dataclass
class SubscriptTableWalker(RatWalker):
    variable_table: VariableTable
    subscript_table: SubscriptTable = field(default_factory=SubscriptTable)
    trace_by_reference: Set[ast.Variable] = field(default_factory=set)

    def walk_Statement(self, node: ast.Statement):
        primary_node = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_node.name]

        # Identify variables we won't know the value of yet -- don't try to
        # trace the values of those but trace any indexing into them
        self.trace_by_reference = set()

        self.walk(node.left)
        self.walk(node.right)

        tracers = self.variable_table.tracers()

        traces = defaultdict(lambda: [])
        for row in primary_variable.itertuples():
            executor = OpportunisticExecutor(tracers, row._asdict(), self.trace_by_reference)
            executor.walk(node)

            for traced_node, value in executor.values.items():
                traces[traced_node].append(value)

        for traced_node, values in traces.items():
            self.subscript_table.insert(traced_node, numpy.array(values))

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.variable_table:
            node_variable = self.variable_table[node.name]
            if node_variable.variable_type != VariableType.DATA:
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


# 1. If the variable on the left appears also on the right, mark this
#   statement to be code-generated with a scan and also make sure that
#   the right hand side is shifted appropriately
@dataclass
class CheckAssignedVariableWalker(NodeWalker):
    variable_table: VariableTable
    msg: str = f"The left hand side of an assignment must be a non-data variable"

    def walk_Statement(self, node: ast.Statement):
        if node.op == "=":
            assigned_name = self.walk(node.left)
            if not assigned_name:
                raise CompileError(self.msg, node.left)
            return assigned_name

    def walk_Variable(self, node: ast.Variable):
        if self.variable_table[node.name].variable_type == VariableType.DATA:
            raise CompileError(self.msg, node)
        else:
            return node.name


# 2. Check that the right hand side of a sampling statement is a
#   function call
@dataclass
class CheckSamplingFunctionWalker(NodeWalker):
    variable_table: VariableTable

    def walk_Statement(self, node: ast.Statement):
        if node.op == "~":
            if not self.walk(node.right):
                raise CompileError(self.msg, node.right)

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
        if self.inside_predicate and self.variable_table[node.name] != VariableType.DATA:
            msg = f"Non-data variables cannot appear in ifelse conditions"
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
    data: Union[pandas.DataFrame, Dict]
    program: ast.Program
    max_trace_iterations: int
    variable_table: VariableTable
    subscript_table: SubscriptTable

    def __init__(self, data: Union[pandas.DataFrame, Dict], program: ast.Program, max_trace_iterations: int):
        self.data = data
        self.program = program
        self.max_trace_iterations = max_trace_iterations

    def _build_variable_table(self):
        """
        Builds the variable table, which holds information for all variables in the model.
        """
        self.variable_table = VariableTable(self.data)

        walker = CreateVariableWalker(self.variable_table)
        walker.walk(self.program)

        # Do a sweep to rename the primary variables as necessary
        walker = RenameWalker(self.variable_table)
        walker.walk(self.program)

        walker = InferSubscriptNameWalker(self.variable_table)
        walker.walk(self.program)

        bind_data_to_functions_walker = BindDataToFunctionsWalker(self.variable_table, self.data)
        bind_data_to_functions_walker.walk(self.program)

        # Trace the program to determine parameter domains and check data domains
        for _ in range(self.max_trace_iterations):
            walker = DomainDiscoveryWalker(self.variable_table)
            walker.walk(self.program)

            found_new_traces = False
            for variable_name, tracer in walker.tracers.items():
                variable = self.variable_table[variable_name]
                found_new_traces |= variable.ingest_new_trace(tracer)

            if not found_new_traces:
                break
        else:
            raise CompileError(f"Unable to resolve subscripts after {self.max_trace_iterations} trace iterations")

        walker = SingleConstraintCheckWalker(self.variable_table)
        walker.walk(self.program)

        # Allocate space for the unconstrained to constrained mapping
        self.variable_table.prepare_unconstrained_parameter_mapping()

    def _build_subscript_table(self):
        walker = SubscriptTableWalker(self.variable_table)
        walker.walk(self.program)
        self.subscript_table = walker.subscript_table

    def _pre_codegen_checks(self):
        walker = CheckAssignedVariableWalker(self.variable_table)
        walker.walk(self.program)

        walker = CheckSamplingFunctionWalker(self.variable_table)
        walker.walk(self.program)

        walker = CheckTransformedParameterOrder()
        walker.walk(self.program)

        walker = IfElsePredicateCheckWalker(self.variable_table)
        walker.walk(self.program)

    def compile(self):
        self._build_variable_table()

        self._build_subscript_table()

        self._pre_codegen_checks()

        return self.variable_table, self.subscript_table
