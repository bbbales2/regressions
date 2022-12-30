import functools
import jax
import numpy
import pandas

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Union, Any, Tuple, Optional

import tatsu

from rat import ast, constraints
from rat.codegen_backends import OpportunisticExecutor, DomainDiscoveryExecutor, ContextStack, CodeExecutor
from rat.exceptions import CompileError, AstException
from rat.trace_table import TraceTable
from rat.variable_table import (
    VariableTable,
    AssignedVariableRecord,
    SampledVariableRecord,
    ConstantVariableRecord,
    DynamicVariableRecord, RecurrentVariableRecord,
)
from rat.walker import RatWalker, NodeWalker, flatten_ast


class NameWalker(RatWalker):
    @staticmethod
    def combine(names: List[str]):
        return functools.reduce(lambda x, y: x + y, filter(None, names))

    def walk_Binary(self, node: ast.Binary):
        return self.combine([self.walk(node.left), self.walk(node.right)])

    def walk_IfElse(self, node: ast.IfElse):
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
        # If all candidate variables don't have subscripts: leftmost variable is primary
        elif all(not candidate.arglist for candidate in candidates):
            candidates[0].prime = True
            return candidates[0]
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
            variables_on_right = [node for node in flatten_ast(node.right) if isinstance(node, tatsu.synth.Variable)]
            if any([node.name == assigned_name for node in variables_on_right]):
                self.variable_table[assigned_name] = RecurrentVariableRecord(**self.variable_table[assigned_name].__dict__)

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.variable_table:
            variable = self.variable_table[node.name]
            if isinstance(variable, SampledVariableRecord):
                self.variable_table[node.name] = AssignedVariableRecord(**self.variable_table[node.name].__dict__)
            elif isinstance(variable, ConstantVariableRecord):
                msg = f"Attempting to assign {variable.name} which is data"
                raise CompileError(msg, node)
            elif isinstance(variable, AssignedVariableRecord):
                msg = f"Attempting to re-assign {variable.name}. Variables can only be assigned once"
                raise CompileError(msg, node)
        else:
            self.variable_table[node.name] = AssignedVariableRecord(node.name, len(node.arglist) if node.arglist else 0)
        return node.name


# Check that only the last subscript of primary variable uses on the right hand side of a recurrence relation
#  have expressions
@dataclass
class CheckRightHandSideRecurrenceRelationWalker(RatWalker):
    variable_table: VariableTable
    primary_name: str = None
    in_primary_arglist: ContextStack[bool] = field(default_factory=lambda: ContextStack(False))

    def walk_Statement(self, node: ast.Statement):
        primary_node = get_primary_ast_variable(node)
        primary_name = primary_node.name
        primary_variable = self.variable_table[primary_name]
        self.primary_name = primary_name
        if isinstance(primary_variable, RecurrentVariableRecord):
            self.walk(node.right)

    def walk_Variable(self, node: ast.Variable):
        if node.arglist is not None:
            if self.in_primary_arglist.peek():
                for arg in node.arglist[:-1]:
                    if not self.walk(arg):
                        raise CompileError()
            elif node.name == self.primary_name:
                with self.in_primary_arglist.push(True):
                    for arg in node.arglist:
                        self.walk(arg)
        return True


# Check that subscripts of primary variables don't include expressions
@dataclass
class CheckPrimaryVariableSubscriptWalker(RatWalker):
    primary_name: str = None
    in_primary_arglist: ContextStack[bool] = field(default_factory=lambda: ContextStack(False))
    msg: str = "Primary variables cannot have expressions in their subscripts"

    def walk_Statement(self, node: ast.Statement):
        self.primary_name = get_primary_ast_variable(node).name
        self.walk(node.left)

    def walk_Variable(self, node: ast.Variable):
        if node.arglist:
            if self.in_primary_arglist.peek():
                for arg in node.arglist:
                    if not self.walk(arg):
                        raise CompileError(self.msg, node)
            else:
                if node.name == self.primary_name:
                    with self.in_primary_arglist.push(True):
                        for arg in node.arglist:
                            self.walk(arg)
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
        subscript_names = _get_subscript_names(node)

        if node.arglist:
            argument_count = len(node.arglist)
        else:
            argument_count = 0

        # Create a variable table record
        if node.name not in self.variable_table:
            if self.in_control_flow.peek():
                if argument_count == 0:
                    if node.name not in self.primary_node_and_subscript_names:
                        msg = f"{node.name} is in control flow/subscripts and must be a primary variable subscript, but it is not"
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
            # Update an existing variable record
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
                    f" been renamed to {primary_variable.subscripts}"
                )
                raise CompileError(msg, primary_ast_variable)


@dataclass
class CheckSubscriptNamesExistWalker(RatWalker):
    variable_table: VariableTable
    primary_node_and_subscript_names: Set[str] = field(default_factory=set)

    def walk_Statement(self, node: ast.Statement):
        primary_node = get_primary_ast_variable(node)
        self.primary_node_and_subscript_names = (
            {primary_node.name} | {node.name for node in primary_node.arglist} if primary_node.arglist else {}
        )

    def walk_Variable(self, node: ast.Variable):
        if node.name in self.primary_node_and_subscript_names:
            return

        variable = self.variable_table[node.name]

        if variable._subscripts is None:
            msg = (
                f"No subscript names found for {node.name}. Variables must either be assigned or inferred. In either"
                "case the subscript names should be specified there"
            )
            raise CompileError(msg, node)

        if node.arglist:
            for arg in node.arglist:
                self.walk(arg)


@dataclass
class DomainDiscoveryWalker(RatWalker):
    variable_table: VariableTable

    def walk_Statement(self, node: ast.Statement):
        # Find the primary variable
        primary_ast_variable = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_ast_variable.name]

        for known_values_as_dict in primary_variable.opportunistic_dict_iterator():
            executor = DomainDiscoveryExecutor(self.variable_table, known_values_as_dict)
            executor.walk(node)


@dataclass
class TraceTableWalker(RatWalker):
    variable_table: VariableTable
    trace_table: TraceTable = field(default_factory=TraceTable)

    def walk_Statement(self, node: ast.Statement):
        primary_node = get_primary_ast_variable(node)
        primary_variable = self.variable_table[primary_node.name]
        recurrent_variable_name = primary_variable.name if isinstance(primary_variable, RecurrentVariableRecord) else None

        self.walk(node.left)
        self.walk(node.right)

        traces = defaultdict(lambda: [])
        for row_number, known_values_as_dict in enumerate(primary_variable.opportunistic_dict_iterator()):
            executor = OpportunisticExecutor(self.variable_table, known_values_as_dict, row_number, recurrent_variable_name)
            executor.walk(node)

            for traced_node, value in executor.values.items():
                traces[traced_node].append((row_number, value))

        dense_size = len(primary_variable)
        for traced_node, trace in traces.items():
            idxs, values = zip(*trace)

            numpy_idxs = numpy.array(idxs)
            numpy_values = numpy.array(values)

            # The "dense" comes from the fact that not every part of an expression will be evaluated
            #  for every element of the primary domain (there could be IfElses). To simplify things,
            #  we pretend as if things are evaluated every time
            dense_values = numpy.zeros(dense_size, dtype=numpy_values.dtype)
            dense_values[numpy_idxs] = numpy_values

            self.trace_table.insert(traced_node, numpy.array(dense_values))


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


# TODO: This check would need to happen at the model level now
# # 4. Parameters cannot be assigned after they are referenced
# @dataclass
# class CheckTransformedParameterOrder(RatWalker):
#     referenced: Set[str] = field(default_factory=set)
#
#     def walk_Statement(self, node: ast.Statement):
#         if node.op == "=":
#             node_name = self.walk(node.left)
#             if node_name is not None:
#                 if node_name in self.referenced:
#                     msg = f"Variable {node_name} cannot be assigned after it is used"
#                     raise CompileError(msg, node.left)
#
#                 self.referenced.add(node_name)
#
#     def walk_Variable(self, node: ast.Variable):
#         return node.name


class StatementComponent:
    statement: ast.Statement
    variable_table: VariableTable
    trace_table: TraceTable
    lower_constraint_node: Optional[ast.ModelBase]
    upper_constraint_node: Optional[ast.ModelBase]

    def __init__(self, statement: ast.Statement, variable_table: VariableTable, data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]):
        walker = CheckPrimaryVariableSubscriptWalker()
        walker.walk(statement)

        walker = CreateAssignedVariablesWalker(variable_table)
        walker.walk(statement)

        walker = CreateVariableWalker(variable_table, data)
        walker.walk(statement)

        walker = RenameSubscriptWalker(variable_table)
        walker.walk(statement)

        walker = CheckSubscriptNamesExistWalker(variable_table)
        walker.walk(statement)

        walker = CheckRightHandSideRecurrenceRelationWalker(variable_table)
        walker.walk(statement)

        # Trace the program to determine parameter domains and check data domains
        walker = DomainDiscoveryWalker(variable_table)
        walker.walk(statement)

        walker = SingleConstraintCheckWalker(variable_table)
        walker.walk(statement)

        primary_node = get_primary_ast_variable(statement)
        self.lower_constraint_node = None
        self.upper_constraint_node = None
        if primary_node.constraints:
            left_node = primary_node.constraints.left
            right_node = primary_node.constraints.right

            left_name = left_node.name
            if right_node:
                right_name = right_node.name
            else:
                right_name = None

            if left_name == "lower" and right_name is None:
                self.lower_constraint_node = left_node.value
                self.upper_constraint_node = None
            elif left_name == "upper" and right_name is None:
                self.lower_constraint_node = None
                self.upper_constraint_node = left_node.value
            elif left_name == "lower" and right_name == "upper":
                self.lower_constraint_node = left_node.value
                self.upper_constraint_node = right_node.value

        walker = TraceTableWalker(variable_table)
        walker.walk(statement)
        trace_table = walker.trace_table

        walker = CheckSamplingFunctionWalker(variable_table)
        walker.walk(statement)

        walker = ControlFlowParameterCheckWalker(variable_table)
        walker.walk(statement)

        self.statement = statement
        self.variable_table = variable_table
        self.trace_table = trace_table

    def evaluate(self, parameters: Dict[str, jax.numpy.ndarray], include_jacobian=False) -> Tuple[Dict[str, jax.numpy.ndarray], float]:
        primary_node = get_primary_ast_variable(self.statement)
        primary_name = primary_node.name
        primary_variable = self.variable_table[primary_name]

        target = 0.0
        traced_keys = sorted(self.trace_table.subscript_dict, key=lambda node: node.text)
        traced_arrays = [self.trace_table.subscript_dict[key].array for key in traced_keys]

        def mapper(node, traced_value):
            executor = CodeExecutor(dict(zip(traced_keys, traced_value)), parameters)
            return executor.walk(node)

        if self.statement.op == "=":
            if self.statement.left.name != primary_name:
                msg = f"Not Implemented Error: The left hand side of assignment must be the primary variable for now"
                raise AstException("computing transformed parameters", msg, self.statement)

            if isinstance(primary_variable, RecurrentVariableRecord):
                primary_variable_nodes_on_right = [
                    node for node in flatten_ast(self.statement.right)
                    if isinstance(node, tatsu.synth.Variable) and node.name == primary_name
                ]

                right_hand_side_traces = {
                    node: self.trace_table[node] for node in primary_variable_nodes_on_right
                    if node in self.trace_table
                }

                if len(right_hand_side_traces) == 0:
                    msg = (
                        "Internal compiler error: Statement compiled as recurrence relation but no primary variable"
                        " references found on right hand side"
                    )
                    raise CompileError(msg, self.statement)

                simplified_traces = {}
                for node, trace in right_hand_side_traces.items():
                    simplified_trace_elements = set(trace.array[trace.array != 0])

                    if len(simplified_trace_elements) == 0:
                        msg = "Internal error: Traced array has zero elements"
                        raise CompileError(msg, node)
                    elif len(simplified_trace_elements) > 1:
                        msg = (
                            "There can only be one offset value for a node on the right hand side of a recurrence"
                            f" found {simplified_trace_elements}"
                        )
                        raise CompileError(msg, node)

                    simplified_traces[node] = simplified_trace_elements.pop()
                carry_size = max(abs(trace) for trace in simplified_traces.values())

                def scanner(carry, traced_value):
                    traced_dict = { **dict(zip(traced_keys, traced_value)), **simplified_traces }
                    executor = CodeExecutor(traced_dict, parameters, primary_name, carry)
                    next_value = executor.walk(self.statement.right)
                    return (next_value,) + carry[:len(carry) - 1], next_value

                _, scanned = jax.lax.scan(scanner, carry_size * (0.0,), traced_arrays)
                parameters[primary_variable.name] = scanned
            else:
                assignment_mapper = functools.partial(mapper, self.statement.right)

                assert primary_variable.name not in parameters

                if primary_variable.argument_count > 0:
                    parameters[primary_variable.name] = jax.vmap(assignment_mapper)(traced_arrays)
                else:
                    parameters[primary_variable.name] = assignment_mapper(traced_arrays)
        elif self.statement.op == "~":
            lower_mapper = functools.partial(mapper, self.lower_constraint_node)
            upper_mapper = functools.partial(mapper, self.upper_constraint_node)
            statement_mapper = functools.partial(mapper, self.statement)

            if primary_variable.argument_count > 0:
                upper = jax.vmap(upper_mapper)(traced_arrays) if self.upper_constraint_node else None
                lower = jax.vmap(lower_mapper)(traced_arrays) if self.lower_constraint_node else None
            else:
                upper = upper_mapper(traced_arrays) if self.upper_constraint_node else None
                lower = lower_mapper(traced_arrays) if self.lower_constraint_node else None

            jacobian_adjustment = 0.0
            if lower is not None or upper is not None:
                unconstrained = parameters[primary_variable.name]

                if lower is not None and upper is None:
                    constrained, jacobian_adjustment = constraints.lower(unconstrained, lower)
                elif lower is None and upper is not None:
                    constrained, jacobian_adjustment = constraints.upper(unconstrained, upper)
                elif lower is not None and upper is not None:  # "lower" in constraints and "upper" in constraints:
                    constrained, jacobian_adjustment = constraints.finite(unconstrained, lower, upper)

                parameters[primary_variable.name] = constrained

            if primary_variable.argument_count > 0:
                target = jax.numpy.sum(jax.vmap(statement_mapper)(traced_arrays)) + (
                    jax.numpy.sum(jacobian_adjustment) if include_jacobian else 0.0
                )
            else:
                target = statement_mapper(traced_arrays) + (jax.numpy.sum(jacobian_adjustment) if include_jacobian else 0.0)
        else:
            msg = f"Unrecognized statement operator {self.statement.op}"
            raise CompileError(msg, self.statement)

        return parameters, target
