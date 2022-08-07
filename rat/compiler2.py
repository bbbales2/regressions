from collections import defaultdict
from dataclasses import dataclass, field
import functools
import numpy
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import List, Dict, Set, Union, Any
import itertools

import pandas as pd

from . import constraints
from . import ast
from .codegen_backends import CodeExecutor, IndentWriter, BaseCodeGenerator, TraceExecutor
from .exceptions import CompileError, MergeError
from .variable_table import Tracer, VariableRecord, VariableTable, VariableType
from .subscript_table import SubscriptTable
from .walker import RatWalker, NodeWalker
from rat import variable_table
from .position_and_range import Position, Range
from rat import subscript_table


@dataclass
class StatementInfo:
    statement: ast.Statement
    primary: ast.Variable


def _get_subscript_names(node: ast.Variable) -> List[str]:
    """
    Examine every subscript argument to see if it has only one named variable

    If this is true for all arguments, return a list with these names
    """
    if node.arglist is None:
        return None

    def combine(names):
        return functools.reduce(lambda x, y: x + y, filter(None, names))

    class NameWalker(RatWalker):
        def walk_Logical(self, node: ast.Logical):
            return combine([self.walk(node.left), self.walk(node.right)])

        def walk_Binary(self, node: ast.Binary):
            return combine([self.walk(node.left), self.walk(node.right)])

        def walk_FunctionCall(self, node: ast.FunctionCall):
            return combine([self.walk(arg) for arg in node.arglist])

        def walk_Variable(self, node: ast.Variable):
            return set([node.name])

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


def _get_primary_ast_variable(statement: ast.Statement) -> ast.Variable:
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
                if self.marked == None:
                    self.marked = node
                else:
                    msg = f"Found two marked primary variables {self.marked.name} and {node.name}. There should only be one"
                    raise CompileError(msg, node)
            else:
                self.candidates.append(node)

    walker = PrimaryWalker()
    walker.walk(statement)
    marked = walker.marked
    candidates = walker.candidates

    if marked != None:
        return marked

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        if len(set(candidate.name for candidate in candidates)) == 1:
            msg = (
                f"No marked primary variable but found multiple references to {candidates[0].name}. One reference should be marked manually"
            )
            raise CompileError(msg, candidates[0])
        else:
            msg = f"No marked primary variable and at least {candidates[0].name} and {candidates[1].name} are candidates. A primary variable should be marked manually"
            raise CompileError(msg, candidates[0])

    if len(candidates) == 0:
        msg = f"No primary variable found on line (this means there are no candidate variables)"
        raise CompileError(msg, statement)


class RatCompiler:
    data: Union[pandas.DataFrame, Dict]
    program: ast.Program
    model_code_string: str
    max_trace_iterations: int
    variable_table: VariableTable
    subscript_table: SubscriptTable
    generated_code: str
    statements: List[StatementInfo]

    def __init__(self, data: Union[pandas.DataFrame, Dict], program: ast.Program, model_code_string: str, max_trace_iterations: int):
        self.data = data
        self.program = program
        self.model_code_string = model_code_string
        self.max_trace_iterations = max_trace_iterations
        self.variable_table = None
        self.subscript_table = None
        self.generated_code = ""
        self.statements = []

    def _identify_primary_symbols(self):
        """
        Generate a StatementInfo for each statement in the program
        """
        for ast_statement in self.program.ast.statements:
            primary = _get_primary_ast_variable(ast_statement)
            self.statements.append(StatementInfo(statement=ast_statement, primary=primary))

        # figure out which symbols will have dataframes
        # has_dataframe = set()
        # for top_expr in self.expr_tree_list:
        #    for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
        #        if isinstance(primeable_symbol, ast.Data) or primeable_symbol.subscript is not None:
        #            has_dataframe.add(primeable_symbol.get_key())

    def _build_variable_table(self):
        """
        Builds the variable table, which holds information for all variables in the model.
        """
        self.variable_table = VariableTable(self.data)

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
                            self.variable_table.insert(
                                variable_name=node.name, argument_count=argument_count, variable_type=VariableType.PARAM
                            )

                if node.arglist:
                    self.in_control_flow = True
                    for arg in node.arglist:
                        self.walk(arg)
                    self.in_control_flow = False

        walker = CreateVariableWalker(self.variable_table)
        walker.walk(self.program)

        # Do a sweep to rename the primary variables as necessary
        for statement_info in self.statements:
            # Find the primary variable
            primary_ast_variable = statement_info.primary
            primary_variable = self.variable_table[primary_ast_variable.name]

            primary_subscript_names = _get_subscript_names(primary_ast_variable)

            if primary_subscript_names:
                # TODO: Make property?
                try:
                    primary_variable.rename(primary_subscript_names)
                except AttributeError:
                    msg = f"Attempting to rename subscripts to {primary_subscript_names} but they have already been renamed to {primary_variable.subscripts}"
                    raise CompileError(msg, primary_ast_variable)

        # Do a sweep to infer subscript names for subscripts not renamed
        for statement_info in self.statements:
            # Find the primary variable
            primary_ast_variable = statement_info.primary
            statement = statement_info.statement
            primary_variable = self.variable_table[primary_ast_variable.name]

            @dataclass
            class InferSubscriptNameWalker(RatWalker):
                primary_name: str
                variable_table: VariableTable

                def walk_Variable(self, node: ast.Variable):
                    if node.name != self.primary_name:
                        if node.arglist:
                            variable = self.variable_table[node.name]
                            subscript_names = _get_subscript_names(node)

                            if variable.variable_type != VariableType.DATA and not variable.renamed and subscript_names is not None:
                                try:
                                    variable.suggest_names(subscript_names)
                                except AttributeError:
                                    msg = f"Attempting to reference subscript of {node.name} as {subscript_names}, but they have already been referenced as {variable.subscripts}. The subscripts must be renamed"
                                    raise CompileError(msg, node)

                    if node.arglist:
                        for arg in node.arglist:
                            self.walk(arg)

            walker = InferSubscriptNameWalker(primary_ast_variable.name, self.variable_table)
            walker.walk(statement)

        # Greedily try to bind variables to data
        @dataclass
        class BindDataToFunctionsWalker(RatWalker):
            variable_table: VariableTable
            data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]

            def walk_Variable(self, node: ast.Variable):
                # Do not bind variables without an arglist
                if node.arglist:
                    subscript_names = _get_subscript_names(node)

                    variable = self.variable_table[node.name]
                    variable.bind(subscript_names, self.data)

                    for arg in node.arglist:
                        self.walk(arg)

        bind_data_to_functions_walker = BindDataToFunctionsWalker(self.variable_table, self.data)
        bind_data_to_functions_walker.walk(self.program)

        # Perform a number of dataframe compatibility checks
        # for statement_info in self.statements:
        #     # Find the primary variable
        #     primary_ast_variable = statement_info.primary
        #     statement = statement_info.statement
        #     primary_variable = self.variable_table[primary_ast_variable.name]
        #     primary_subscript_names = primary_variable.subscripts

        #     data_names = self.get_data_names() | set(primary_subscript_names)

        #     @dataclass
        #     class DataframeCompatibilityWalker(RatWalker):
        #         primary_name: str
        #         data_names: List[str]
        #         variable_table: VariableTable

        #         def walk_Variable(self, node: ast.Variable):
        #             # TODO: It would be nice if these code paths were closer for Data and Params
        #             variable = self.variable_table[node.name]
        #             primary_variable = self.variable_table[self.primary_name]

        #             subscript_names = _get_subscript_names(node)
        #             primary_subscript_names = primary_variable.subscripts

        #             if node.arglist:
        #                 # Check that number/names of subscripts are compatible with primary variable
        #                 for name, arg in zip(subscript_names, node.arglist):
        #                     if name not in primary_subscript_names:
        #                         dataframe_name = self.variable_table.get_dataframe_name(self.primary_name)
        #                         msg = f"Subscript {name} not found in dataframe {dataframe_name} (associated with primary variable {self.primary_name})"
        #                         raise CompileError(msg, arg)

        #             if node.name in self.data_names:
        #                 # If the target variable is data, then assume the subscripts here are
        #                 # referencing columns of the dataframe by name
        #                 if node.arglist:
        #                     # Check that the names of subscripts are compatible with the variable's own dataframe
        #                     for name, arg in zip(subscript_names, node.arglist):
        #                         if name not in variable.subscripts:
        #                             dataframe_name = self.variable_table.get_dataframe_name(node.name)
        #                             msg = f"Subscript {name} not found in dataframe {dataframe_name} (associated with variable {node.name})"
        #                             raise CompileError(msg, arg)
        #             else:
        #                 # If variable has no subscript, it's a scalar and there's nothing to do
        #                 if subscript_names is not None:
        #                     # If variable is known to have subscripts then check that
        #                     # the number of subscripts are compatible with existing
        #                     # ones
        #                     if len(variable.subscripts) > 0:
        #                         if len(subscript_names) != len(variable.subscripts):
        #                             msg = f"{len(subscript_names)} found, previously used with {len(variable.subscripts)}"
        #                             raise CompileError(msg, node)

        #                     # Fold all the shift expressions to simple integers
        #                     # TODO: Is this necessary?
        #                     # shifts, pad_needed = self._get_shifts_and_padding(symbol.subscript)
        #                     # variable.pad_needed = pad_needed

        #                     # Extra checks for secondary variables
        #                     # TODO: Is this necessary?
        #                     # if symbol_key != primary_symbol_key:
        #                     # TODO: This should probably be supported in the future
        #                     # right now I'm leaving it off because I'm not quite sure
        #                     # what the behavior should be and I don't have an example
        #                     # model in mind (my guess is access should give zeros)
        #                     #     if any(shift != 0 for shift in shifts):
        #                     #         msg = f"Shifted access on a secondary parameter is not allowed"
        #                     #         raise CompileError(msg, symbol.range)

        #     walker = DataframeCompatibilityWalker(primary_ast_variable.name, data_names, self.variable_table)
        #     walker.walk(statement)

        # Trace the program to determine parameter domains and check data domains
        for _ in range(self.max_trace_iterations):
            # Reset all the parameter tracers
            tracers = {}
            for name in self.variable_table:
                variable = self.variable_table[name]
                if not variable.tracer:
                    variable.tracer = Tracer()
                tracers[name] = variable.tracer.copy()

            for statement_info in self.statements:
                # Find the primary variable
                primary_ast_variable = statement_info.primary
                primary_variable = self.variable_table[primary_ast_variable.name]

                for row in primary_variable.itertuples():
                    executor = TraceExecutor(tracers, row._asdict())
                    executor.walk(statement_info.statement)

            found_new_traces = False
            for variable_name, tracer in tracers.items():
                variable = self.variable_table[variable_name]
                found_new_traces |= variable.ingest_new_trace(tracer)

            if not found_new_traces:
                break
        else:
            raise CompileError(f"Unable to resolve subscripts after {self.max_trace_iterations} trace iterations")

        # Apply constraints to parameters
        @dataclass
        class ConstraintWalker(RatWalker):
            variable_table: VariableTable

            def walk_Variable(self, node: ast.Variable):
                variable = self.variable_table[node.name]
                if variable.variable_type == VariableType.PARAM and node.constraints is not None:
                    # Constraints should be evaluated at compile time
                    codegen = BaseCodeGenerator()
                    try:
                        lower_constraint_value = float("-inf")
                        upper_constraint_value = float("inf")

                        left_constraint_name = node.constraints.left.name
                        left_constraint_value = float(eval(codegen.walk(node.constraints.left.value)))

                        if left_constraint_name == "lower":
                            lower_constraint_value = left_constraint_value
                        else:
                            upper_constraint_value = left_constraint_value

                        if node.constraints.right is not None:
                            right_constraint_name = node.constraints.right.name
                            right_constraint_value = float(eval(codegen.walk(node.constraints.right.value)))

                            if right_constraint_name == "lower":
                                lower_constraint_value = right_constraint_value
                            else:
                                upper_constraint_value = right_constraint_value

                    except Exception as e:
                        error_msg = f"Failed evaluating constraints for parameter {node.name}, ({e})"
                        raise CompileError(error_msg, node) from e

                    try:
                        variable.set_constraints(lower_constraint_value, upper_constraint_value)
                    except AttributeError:
                        msg = f"Attempting to set constraints of {node.name} to ({lower_constraint_value}, {upper_constraint_value}) but they are already set to ({variable.constraint_lower}, {variable.constraint_upper})"
                        raise CompileError(msg, node)

        walker = ConstraintWalker(self.variable_table)
        walker.walk(self.program)

        # Allocate space for the unconstrained to constrained mapping
        self.variable_table.prepare_unconstrained_parameter_mapping()

    def _pre_codegen_checks(self):
        N_lines = len(self.statements)

        @dataclass
        class FindAndExplodeWalker(RatWalker):
            search_name: str
            error_msg: str

            def walk_Variable(self, node: ast.Variable):
                if self.search_name == node.name:
                    raise CompileError(self.error_msg, node)

        for line_index, statement_info in enumerate(self.statements):
            primary_name = statement_info.primary.name
            primary_variable = self.variable_table[primary_name]
            statement = statement_info.statement
            assigned_name = None

            # 1. If the variable on the left appears also on the right, mark this
            #   statement to be code-gen'd with a scan and also make sure that
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

            walker = CheckAssignedVariableWalker(self.variable_table)
            assigned_name = walker.walk(statement)

            # 2. Check that the right hand side of a sampling statement is a
            #   function call
            @dataclass
            class CheckSamplingFunctionWalker(NodeWalker):
                variable_table: VariableTable
                msg: str = f"The right hand side of a sampling statement must be a function"

                def walk_Statement(self, node: ast.Statement):
                    if node.op == "~":
                        if not self.walk(node.right):
                            raise CompileError(self.msg, node.right)

                def walk_FunctionCall(self, node: ast.FunctionCall):
                    return True

            walker = CheckSamplingFunctionWalker(self.variable_table)
            walker.walk(statement)

            # 3. Check that all secondary parameter uses precede primary uses (or throw an error)
            if primary_variable.variable_type != VariableType.DATA:
                msg = f"Primary variable {primary_name} used on line {line_index} but then referenced as a non-prime variable. The primed uses must come last"
                walker = FindAndExplodeWalker(primary_name, msg)
                for j in range(line_index + 1, N_lines):
                    following_statement_info = self.statements[j]
                    # Throw an error for any subsequent non-primary uses of the primary variable
                    if following_statement_info.primary.name != primary_name:
                        walker.walk(following_statement_info.statement)

            # 4. Parameters cannot be used after they are assigned
            if statement.op == "=":
                msg = f"Parameter {assigned_name} is assigned on line {line_index} but used after. A variable cannot be used after it is assigned"
                walker = FindAndExplodeWalker(assigned_name, msg)
                for j in range(line_index + 1, N_lines):
                    walker.walk(self.statements[j].statement)

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

        walker = IfElsePredicateCheckWalker(self.variable_table)
        walker.walk(self.program)

    def _build_subscript_table(self):
        @dataclass
        class SubscriptTableWalker(RatWalker):
            variable_table: VariableTable
            subscript_table: SubscriptTable = field(default_factory=SubscriptTable)
            trace_by_reference: Set[ast.Variable] = field(default_factory=set)

            def walk_Statement(self, node: ast.Statement):
                primary_node = _get_primary_ast_variable(node)
                primary_variable = self.variable_table[primary_node.name]

                # Identify variables we won't know the value of yet -- don't try to
                # trace the values of those but trace any indexing into them
                self.trace_by_reference = set()

                self.walk(node.left)
                self.walk(node.right)

                tracers = self.variable_table.tracers()

                traces = defaultdict(lambda : [])
                for row in primary_variable.itertuples():
                    executor = TraceExecutor(tracers, row._asdict(), self.trace_by_reference)
                    executor.walk(node)

                    for traced_node, value in executor.first_level_values.items():
                        traces[traced_node].append(value)
                
                for traced_node, values in traces.items():
                    self.subscript_table.insert(traced_node, numpy.array(values))

            def walk_Variable(self, node: ast.Variable):
                if node.name in self.variable_table:
                    node_variable = self.variable_table[node.name]
                    if node_variable.variable_type != VariableType.DATA:
                        self.trace_by_reference.add(node)

        walker = SubscriptTableWalker(self.variable_table)
        walker.walk(self.program)
        self.subscript_table = walker.subscript_table

    def compile(self):
        self._identify_primary_symbols()

        self._build_variable_table()

        self._build_subscript_table()

        self._pre_codegen_checks()

        return self.variable_table, self.subscript_table

@dataclass
class TransformedParametersFunctionGenerator(RatWalker):
    variable_table: VariableTable
    subscript_table: SubscriptTable
    parameters: Dict[str, Any]
    traced_nodes: List[ast.Variable] = None

    def walk_Program(self, node: ast.Program):
        # TODO -- there is some order required here!
        for statement in node.statements:
            self.walk(statement)

    def walk_Statement(self, node: ast.Statement):
        if node.op != "=":
            return

        primary_node = _get_primary_ast_variable(node)
        primary_name = primary_node.name
        primary_variable = self.variable_table[primary_name]

        if node.left.name != primary_name:
            msg = f"Not Implemented Error: The left hand side of assignment must be the primary variable for now"
            raise CompileError(msg, node)
        
        self.traced_nodes = []
        
        self.walk(node.left)
        self.walk(node.right)

        arguments = tuple(self.subscript_table[traced_node].array for traced_node in self.traced_nodes)

        def mapper(*arguments):
            walker = CodeExecutor(dict(zip(self.traced_nodes, arguments)), self.parameters)
            return walker.walk(node.right)

        # We can't be overwriting parameters
        assert primary_name not in self.parameters

        if primary_variable.argument_count > 0:
            self.parameters[primary_name] = jax.vmap(mapper)(*arguments)
        else:
            self.parameters[primary_name] = mapper(*arguments)

    def walk_Variable(self, node: ast.Variable):
        node_variable = self.variable_table[node.name]
        if node_variable.argument_count > 0:
            if node in self.subscript_table:
                self.traced_nodes.append(node)

@dataclass
class EvaluateDensityWalker(RatWalker):
    variable_table: VariableTable
    subscript_table: SubscriptTable
    parameters: Dict[str, Any]

    def walk_Program(self, node: ast.Program):
        target = 0.0
        for statement in node.statements:
            target += self.walk(statement)
        return target

    def walk_Statement(self, node: ast.Statement):
        if node.op != "~":
            return 0.0

        primary_node = _get_primary_ast_variable(node)
        primary_name = primary_node.name
        primary_variable = self.variable_table[primary_name]

        self.traced_nodes = []
        
        self.walk(node.left)
        self.walk(node.right)

        arguments = tuple(self.subscript_table[traced_node].array for traced_node in self.traced_nodes)

        def mapper(*arguments):
            walker = CodeExecutor(dict(zip(self.traced_nodes, arguments)), self.parameters)
            return walker.walk(node)

        if primary_variable.argument_count > 0:
            return jax.numpy.sum(jax.vmap(mapper)(*arguments))
        else:
            return(mapper(*arguments))

    def walk_Variable(self, node: ast.Variable):
        node_variable = self.variable_table[node.name]
        if node_variable.argument_count > 0:
            if node in self.subscript_table:
                self.traced_nodes.append(node)

def compile_for_jax(program : ast.Program, variable_table: VariableTable, subscript_table: SubscriptTable):
    def constrain_function(unconstrained_parameter_vector : numpy.ndarray):
        jacobian, parameters = constrain(variable_table, unconstrained_parameter_vector)

        transform = TransformedParametersFunctionGenerator(variable_table, subscript_table, parameters)
        transform.walk(program)

        return jacobian, parameters

    def target_function(include_jacobian : bool, unconstrained_parameter_vector : numpy.ndarray):
        jacobian, parameters = constrain(variable_table, unconstrained_parameter_vector)

        # Modify parameters in place
        transform = TransformedParametersFunctionGenerator(variable_table, subscript_table, parameters)
        transform.walk(program)

        likelihood = EvaluateDensityWalker(variable_table, subscript_table, parameters)
        target = likelihood.walk(program)

        return target + (jacobian if include_jacobian else 0.0)

    return constrain_function, target_function

def constrain(variable_table : VariableTable, unconstrained_parameter_vector : numpy.ndarray):
    parameters = {}
    jacobian_adjustments = 0.0

    for name in variable_table:
        record = variable_table[name]

        if record.variable_type != VariableType.PARAM:
            continue

        # This assumes that unconstrained parameter indices for a parameter is allocated in a contiguous fashion.
        if len(record.subscripts) > 0:
            unconstrained = unconstrained_parameter_vector[record.unconstrained_vector_start_index : record.unconstrained_vector_end_index + 1]
        else:
            unconstrained = unconstrained_parameter_vector[record.unconstrained_vector_start_index]

        if record.constraint_lower > float("-inf") or record.constraint_upper < float("inf"):
            if record.constraint_lower > float("-inf") and record.constraint_upper == float("inf"):
                constrained, jacobian_adjustment = constraints.lower(unconstrained, record.constraint_lower)
            elif record.constraint_lower == float("inf") and record.constraint_upper < float("inf"):
                constrained, jacobian_adjustment = constraints.upper(unconstrained, record.constraint_upper)
            elif record.constraint_lower > float("-inf") and record.constraint_upper < float("inf"):
                constrained, jacobian_adjustment = constraints.finite(unconstrained, record.constraint_lower, record.constraint_upper)

            jacobian_adjustments += jax.numpy.sum(jacobian_adjustment)
        else:
            constrained = unconstrained

        parameters[name] = constrained

    return jacobian_adjustments, parameters

