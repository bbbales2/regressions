from dataclasses import dataclass, field
import functools
import pandas
import jax
import jax.numpy
import jax.scipy.stats
from typing import List, Dict, Set, Union
import itertools

import pandas as pd

from . import ast2
from . import codegen_backends2
from .exceptions2 import CompileError, MergeError
from .variable_table import VariableTracer, VariableTable, VariableType
from .walker import RatWalker
from rat import variable_table
from .position_and_range import Position, Range


@dataclass
class StatementInfo:
    statement: ast2.Statement
    primary: ast2.Variable


def _get_subscript_names(node: ast2.Variable) -> List[str]:
    """
    Examine every subscript argument to see if it has only one named variable

    If this is true for all arguments, return a list with these names
    """
    if node.arglist is None:
        return None

    def combine(names):
        return functools.reduce(lambda x, y: x + y, filter(None, names))

    class NameWalker(RatWalker):
        def walk_Logical(self, node: ast2.Logical):
            return combine([self.walk(node.left), self.walk(node.right)])

        def walk_Binary(self, node: ast2.Binary):
            return combine([self.walk(node.left), self.walk(node.right)])

        def walk_FunctionCall(self, node: ast2.FunctionCall):
            return combine([self.walk(arg) for arg in node.arglist])

        def walk_Variable(self, node: ast2.Variable):
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


class RatCompiler:
    data: Union[pandas.DataFrame, Dict]
    program: ast2.Program
    model_code_string: str
    max_trace_iterations: int
    variable_table: VariableTable
    generated_code: str
    statements: List[StatementInfo]

    def __init__(self, data: Union[pandas.DataFrame, Dict], program: ast2.Program, model_code_string: str, max_trace_iterations: int):
        self.data = data
        self.program = program
        self.model_code_string = model_code_string
        self.max_trace_iterations = max_trace_iterations
        self.variable_table: VariableTable = None
        self.generated_code = ""
        self.statements = []

    def get_data_names(self) -> Set[str]:
        """
        Get the names of all the input dataframe columns
        """
        match self.data:
            case pandas.DataFrame():
                return set(self.data.columns)
            case _:
                names = set()
                for df in self.data:
                    names |= set(df.columns)
                return names

    def _get_primary_ast_variable(self, statement: ast2.Statement):
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
            marked: ast2.Variable = None
            candidates: List[ast2.Variable] = field(default_factory=list)

            def walk_Variable(self, node: ast2.Variable):
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
                msg = f"No marked primary variable but found multiple references to {candidates[0].name}. One reference should be marked manually"
                raise CompileError(msg, candidates[0])
            else:
                msg = f"No marked primary variable and at least {candidates[0].name} and {candidates[1].name} are candidates. A primary variable should be marked manually"
                raise CompileError(msg, candidates[0])

        if len(candidates) == 0:
            msg = f"No primary variable found on line (this means there are no candidate variables)"
            raise CompileError(msg, statement)

    def _identify_primary_symbols(self):
        """
        Generate a StatementInfo for each statement in the program
        """
        for ast_statement in self.program.ast.statements:
            primary = self._get_primary_ast_variable(ast_statement)
            self.statements.append(StatementInfo(statement=ast_statement, primary=primary))

        # figure out which symbols will have dataframes
        # has_dataframe = set()
        # for top_expr in self.expr_tree_list:
        #    for primeable_symbol in ast.search_tree(top_expr, ast.PrimeableExpr):
        #        if isinstance(primeable_symbol, ast.Data) or primeable_symbol.subscript is not None:
        #            has_dataframe.add(primeable_symbol.get_key())

    def build_variable_table(self):
        """
        Builds the variable table, which holds information for all variables in the model.
        """
        self.variable_table = VariableTable(self.data)

        # Add entries to the variable table for all ast variables
        @dataclass
        class CreateVariableWalker(RatWalker):
            data_names: Set[str]
            variable_table: VariableTable
            left_hand_of_assignment: bool = False

            def walk_Statement(self, node: ast2.Statement):
                if node.op == "=":
                    old_left_hand_of_assignment = self.left_hand_of_assignment
                    self.left_hand_of_assignment = True
                    self.walk(node.left)
                    self.left_hand_of_assignment = old_left_hand_of_assignment
                else:
                    self.walk(node.left)
                self.walk(node.right)

            def walk_Variable(self, node: ast2.Variable):
                if node.name in data_names:
                    self.variable_table.insert(variable_name=node.name, variable_type=VariableType.DATA)
                else:
                    # Overwrite table entry for assigned parameters (so they don't get turned back into regular parameters)
                    if self.left_hand_of_assignment:
                        self.variable_table.insert(variable_name=node.name, variable_type=VariableType.ASSIGNED_PARAM)
                    else:
                        if node.name not in self.variable_table:
                            self.variable_table.insert(variable_name=node.name, variable_type=VariableType.PARAM)

        data_names = self.get_data_names()
        walker = CreateVariableWalker(data_names, self.variable_table)
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

                def walk_Variable(self, node: ast2.Variable):
                    if node.name != self.primary_name:
                        variable = self.variable_table[node.name]
                        subscript_names = _get_subscript_names(node)

                        if not variable.renamed and subscript_names is not None:
                            try:
                                variable.suggest_names(subscript_names)
                            except AttributeError:
                                msg = f"Attempting to reference subscript of {node.name} as {subscript_names}, but they have already been referenced as {variable.subscripts}. The subscripts must be renamed"
                                raise CompileError(msg, node)

            walker = InferSubscriptNameWalker(primary_ast_variable.name, self.variable_table)
            walker.walk(statement)

        # Perform a number of dataframe compatibility checks
        for statement_info in self.statements:
            # Find the primary variable
            primary_ast_variable = statement_info.primary
            statement = statement_info.statement
            primary_variable = self.variable_table[primary_ast_variable.name]
            primary_subscript_names = primary_variable.subscripts

            data_names = self.get_data_names() | set(primary_subscript_names)

            @dataclass
            class DataframeCompatibilityWalker(RatWalker):
                primary_name: str
                data_names: List[str]
                variable_table: VariableTable

                def walk_Variable(self, node: ast2.Variable):
                    # TODO: It would be nice if these code paths were closer for Data and Params
                    variable = self.variable_table[node.name]
                    primary_variable = self.variable_table[self.primary_name]

                    subscript_names = _get_subscript_names(node)
                    primary_subscript_names = primary_variable.subscripts

                    if subscript_names is not None:
                        # Check that number/names of subscripts are compatible with primary variable
                        for name, arg in zip(subscript_names, node.arglist):
                            if name not in primary_subscript_names:
                                dataframe_name = self.variable_table.get_dataframe_name(self.primary_name)
                                msg = f"Subscript {name} not found in dataframe {dataframe_name} (associated with primary variable {self.primary_name})"
                                raise CompileError(msg, arg)

                    if node.name in self.data_names:
                        # If the target variable is data, then assume the subscripts here are
                        # referencing columns of the dataframe by name
                        if subscript_names is not None:
                            # Check that the names of subscripts are compatible with the variable's own dataframe
                            for name, arg in zip(subscript_names, node.arglist):
                                if name not in variable.subscripts:
                                    dataframe_name = self.variable_table.get_dataframe_name(node.name)
                                    msg = (
                                        f"Subscript {name} not found in dataframe {dataframe_name} (associated with variable {symbol_key})"
                                    )
                                    raise CompileError(msg, arg)
                    else:
                        subscript_names = _get_subscript_names(node)

                        # If variable has no subscript, it's a scalar and there's nothing to do
                        if subscript_names is not None:
                            # If variable is known to have subscripts then check that
                            # the number of subscripts are compatible with existing
                            # ones
                            if len(variable.subscripts) > 0:
                                if len(subscript_names) != len(variable.subscripts):
                                    msg = f"{len(subscript_names)} found, previously used with {len(variable.subscripts)}"
                                    raise CompileError(msg, node)

                            # Fold all the shift expressions to simple integers
                            # TODO: Is this necessary?
                            # shifts, pad_needed = self._get_shifts_and_padding(symbol.subscript)
                            # variable.pad_needed = pad_needed

                            # Extra checks for secondary variables
                            # TODO: Is this necessary?
                            # if symbol_key != primary_symbol_key:
                            # TODO: This should probably be supported in the future
                            # right now I'm leaving it off because I'm not quite sure
                            # what the behavior should be and I don't have an example
                            # model in mind (my guess is access should give zeros)
                            #     if any(shift != 0 for shift in shifts):
                            #         msg = f"Shifted access on a secondary parameter is not allowed"
                            #         raise CompileError(msg, symbol.range)

            walker = DataframeCompatibilityWalker(primary_ast_variable.name, data_names, self.variable_table)
            walker.walk(statement)

        # Trace the program to determine parameter domains and check data domains
        tracers = {}

        for _ in range(self.max_trace_iterations):
            # Reset all the parameter tracers
            for variable_name in self.variable_table:
                variable = self.variable_table[variable_name]
                tracers[variable_name] = variable.get_tracer()

            for statement_info in self.statements:
                # Find the primary variable
                primary_ast_variable = statement_info.primary
                primary_variable = self.variable_table[primary_ast_variable.name]

                primary_df = primary_variable.base_df
                if primary_df is not None:
                    code_generator = codegen_backends2.DiscoverVariablesCodeGenerator()
                    code = code_generator.walk(statement_info.statement)

                    for row in primary_df.itertuples(index=False):
                        lambda_row = {key: functools.partial(lambda x: x, value) for key, value in row._asdict().items()}
                        eval(code, globals(), {**tracers, **lambda_row})

            found_new_traces = False
            for variable_name, tracer in tracers.items():
                variable = self.variable_table[variable_name]
                if variable.variable_type != VariableType.DATA:
                    found_new_traces |= variable.ingest_new_trace(tracer)

            if not found_new_traces:
                break
        else:
            raise CompileError(f"Unable to resolve subscripts after {self.max_trace_iterations} trace iterations")

        # Apply constraints to parameters
        @dataclass
        class ConstraintWalker(RatWalker):
            variable_table: VariableTable

            def walk_Variable(self, node: ast2.Variable):
                variable = self.variable_table[node.name]
                if variable.variable_type == VariableType.PARAM and node.constraints is not None:
                    # Constraints should be evaluated at compile time
                    codegen = codegen_backends2.BaseCodeGenerator()
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

    def pre_codegen_checks(self):
        N_lines = len(self.statements)

        @dataclass
        class FindAndExplodeWalker(RatWalker):
            search_name: str
            error_msg: str

            def walk_Variable(self, node: ast2.Variable):
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
            if statement.op == "=":
                lhs = statement.left
                msg = f"The left hand side of an assignment must be a non-data variable"
                if not isinstance(lhs, ast2.Variable):
                    raise CompileError(msg, lhs)

                assigned_name = lhs.name

                if self.variable_table[lhs.name].variable_type == VariableType.DATA:
                    raise CompileError(msg, lhs)

            #         # The left hand side is written by the assignment but will be updated
            #         # after the scan is complete and should not be marked
            #         for symbol in ast.search_tree(rhs, ast.Param):
            #             symbol_key = symbol.get_key()

            #             if assigned_key == symbol_key:
            #                 symbol.assigned_by_scan = True

            #                 if symbol.subscript is None or all(shift is None for shift in symbol.subscript.shifts):
            #                     msg = f"Recursively assigning {symbol_key} requires a shifted subscript on the right hand side reference"
            #                     raise CompileError(msg, symbol.range)

            #                 shifts = [shift.value for shift in symbol.subscript.shifts]

            #                 if sum(shift != 0 for shift in shifts) != 1:
            #                     msg = "Exactly one (no more, no less) subscripts can be shifted in a recursively assigned variable"
            #                     raise CompileError(msg, symbol.range)

            #                 if shifts[-1] <= 0:
            #                     msg = "Only the right-most subscript of a recursively assigned variable can be shifted"
            #                     raise CompileError(msg, symbol.range)

            #                 if any(shift < 0 for shift in shifts):
            #                     msg = "All nonzero shifts in a recursively assigned variable must be positive"
            #                     raise CompileError(msg, symbol.range)

            # 2. Check that all secondary parameter uses precede primary uses (or throw an error)
            if primary_variable.variable_type != VariableType.DATA:
                msg = f"Primary variable {primary_name} used on line {line_index} but then referenced as a non-prime variable. The primed uses must come last"
                walker = FindAndExplodeWalker(primary_name, msg)
                for j in range(line_index + 1, N_lines):
                    following_statement_info = self.statements[j]
                    # Throw an error for any subsequent non-primary uses of the primary variable
                    if following_statement_info.primary.name != primary_name:
                        walker.walk(following_statement_info.statement)

            # 3. Parameters cannot be used after they are assigned
            if statement_info.statement == "=":
                msg = f"Parameter {assigned_name} is assigned on line {line_index} but used after. A variable cannot be used after it is assigned"
                walker = FindAndExplodeWalker(assigned_name, msg)
                for j in range(line_index + 1, N_lines):
                    walker.walk(self.statements[j].statement)

        @dataclass
        class IfElsePredicateCheckWalker(RatWalker):
            variable_table: VariableTable
            inside_predicate: bool = False

            def walk_IfElse(self, node: ast2.IfElse):
                old_inside_predicate = self.inside_predicate
                self.inside_predicate = True
                self.walk(node.predicate)
                self.inside_predicate = old_inside_predicate

                self.walk(node.left)
                self.walk(node.right)

            def walk_Variable(self, node: ast2.Variable):
                if self.inside_predicate and self.variable_table[node.name] != VariableType.DATA:
                    msg = f"Non-data variables cannot appear in ifelse conditions"
                    raise CompileError(self.error_msg, node)

        walker = IfElsePredicateCheckWalker(self.variable_table)
        walker.walk(self.program)

    def codegen(self):
        self.generated_code = ""
        self.generated_code += "# A rat model\n\n"
        self.generated_code += "import rat.constraints\n"
        self.generated_code += "import rat.math\n"
        self.generated_code += "import jax\n\n"

        self.generated_code += f"unconstrained_parameter_size = {self.variable_table.unconstrained_parameter_size}\n\n"

        self.codegen_constrain_parameters()

        self.codegen_transform_parameters()

        self.codegen_evaluate_densities()

    def codegen_constrain_parameters(self):
        self.generated_code += "def constrain_parameters(unconstrained_parameter_vector, pad=True):\n"
        self.generated_code += "    unconstrained_parameters = {}\n"
        self.generated_code += "    parameters = {}\n"
        self.generated_code += "    jacobian_adjustments = 0.0\n"

        for variable_name in self.variable_table:
            record = self.variable_table[variable_name]
            if record.variable_type != VariableType.PARAM:
                continue

            unconstrained_reference = f"unconstrained_parameters['{variable_name}']"
            constrained_reference = f"parameters['{variable_name}']"

            self.generated_code += "\n"
            self.generated_code += f"    # Param: {variable_name}, lower: {record.constraint_lower}, upper: {record.constraint_upper}\n"

            # This assumes that unconstrained parameter indices for a parameter is allocated in a contiguous fashion.
            if len(record.subscripts) > 0:
                index_string = f"{record.unconstrained_vector_start_index} : {record.unconstrained_vector_end_index + 1}"
            else:
                index_string = f"{record.unconstrained_vector_start_index}"

            self.generated_code += f"    {unconstrained_reference} = unconstrained_parameter_vector[..., {index_string}]\n"

            if record.constraint_lower > float("-inf") or record.constraint_upper < float("inf"):
                if record.constraint_lower > float("-inf") and record.constraint_upper == float("inf"):
                    self.generated_code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.lower({unconstrained_reference}, {record.constraint_lower})\n"
                elif record.constraint_lower == float("inf") and record.constraint_upper < float("inf"):
                    self.generated_code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.upper({unconstrained_reference}, {record.constraint_upper})\n"
                elif record.constraint_lower > float("-inf") and record.constraint_upper < float("inf"):
                    self.generated_code += f"    {constrained_reference}, constraints_jacobian_adjustment = rat.constraints.finite({unconstrained_reference}, {record.constraint_lower}, {record.constraint_upper})\n"

                self.generated_code += "    jacobian_adjustments += jax.numpy.sum(constraints_jacobian_adjustment)\n"
            else:
                self.generated_code += f"    {constrained_reference} = {unconstrained_reference}\n"

            if record.pad_needed:
                self.generated_code += "    if pad:\n"
                self.generated_code += f"        {constrained_reference} = jax.numpy.pad({constrained_reference}, (0, 1))\n"

        self.generated_code += "\n"
        self.generated_code += "    return jacobian_adjustments, parameters\n\n"

    def codegen_transform_parameters(self):
        self.generated_code += "def transform_parameters(data, subscripts, first_in_group_indicators, parameters):\n"

        for top_expr in list(reversed(self.expr_tree_list)):
            if not isinstance(top_expr, ast.Assignment):
                continue

            code_generator = codegen_backends.TransformedParametersCodeGenerator(
                self.variable_table, self._get_primary_symbol_from_statement(top_expr), indent=4
            )

            code_generator.generate(top_expr)
            self.generated_code += code_generator.get_expression_string()
            self.generated_code += "\n"

        self.generated_code += "\n"
        self.generated_code += "    return parameters\n\n"

    def codegen_evaluate_densities(self):
        self.generated_code += "def evaluate_densities(data, subscripts, parameters):\n"
        self.generated_code += "    target = 0.0\n"

        for top_expr in self.expr_tree_list:
            if not isinstance(top_expr, ast.Distr):
                continue

            code_generator = codegen_backends.EvaluateDensityCodeGenerator(
                self.variable_table, self._get_primary_symbol_from_statement(top_expr)
            )
            try:
                code_generator.generate(top_expr)
            except MergeError as e:
                msg = str(e)
                raise CompileError(msg, top_expr.range)
            self.generated_code += f"    target += jax.numpy.sum({code_generator.get_expression_string()})\n"

        self.generated_code += "\n"
        self.generated_code += "    return target\n"

    def compile(self):
        self._identify_primary_symbols()

        self.build_variable_table()

        self.variable_table.prepare_unconstrained_parameter_mapping()

        self.pre_codegen_checks()

        self.codegen()

        data_dict = {}
        base_df_dict = {}

        for variable_name in self.variable_table:
            record = self.variable_table[variable_name]
            for data_name, array in record.get_numpy_arrays():
                data_dict[data_name] = jax.numpy.array(array)
            if record.variable_type != VariableType.DATA:
                if record.base_df is not None:
                    base_df_dict[variable_name] = record.base_df
                else:
                    base_df_dict[variable_name] = pd.DataFrame()

        return (
            data_dict,
            base_df_dict,
            self.variable_table.generated_subscript_dict,
            self.variable_table.first_in_group_indicator,
            self.generated_code,
        )
