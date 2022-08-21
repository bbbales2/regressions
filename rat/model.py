from dataclasses import dataclass, field
from functools import partial
import jax
import jax.experimental.host_callback
import numpy
import pandas
from typing import Dict, Union, List, Any
from tatsu.model import ModelBuilderSemantics
from tatsu.walkers import NodeWalker

from .compiler import RatCompiler
from .exceptions import AstException
from .parser import RatParser
from .trace_table import TraceTable
from .variable_table import VariableTable, SampledVariableRecord, DynamicVariableRecord
from . import ast
from . import compiler
from . import constraints
from . import math
from . import walker


class ExecuteException(AstException):
    def __init__(self, message: str, node: ast.ModelBase):
        super().__init__("evaluating log density", message, node)


class CodeExecutor(NodeWalker):
    arguments: Dict[ast.Variable, Any]
    parameters: Dict[str, Any]
    left_side_of_sampling: Union[None, ast.ModelBase]

    def __init__(self, arguments: Dict[ast.Variable, Any] = None, parameters: Dict[str, Any] = None):
        self.arguments = arguments
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
        predicate = self.arguments[node.predicate]
        return jax.lax.cond(predicate, lambda: self.walk(node.left), lambda: self.walk(node.right))

    def walk_FunctionCall(self, node: ast.FunctionCall):
        argument_list = []

        if self.left_side_of_sampling:
            argument_list += [self.walk(self.left_side_of_sampling)]

        if node.arglist:
            argument_list += [self.walk(arg) for arg in node.arglist]

        return getattr(math, node.name)(*argument_list)

    def walk_Variable(self, node: ast.Variable):
        if node in self.arguments:
            if node.name in self.parameters:
                trace = self.arguments[node]
                return self.parameters[node.name][trace]
            else:
                return self.arguments[node]
        else:
            return self.parameters[node.name]

    def walk_Literal(self, node: ast.Literal):
        return node.value


@dataclass
class TransformedParametersFunctionGenerator(walker.RatWalker):
    variable_table: VariableTable
    trace_table: TraceTable
    parameters: Dict[str, Any]
    traced_nodes: List[ast.Variable] = field(default_factory=list)

    def walk_Program(self, node: ast.Program):
        # TODO -- there is some order required here!
        for statement in node.statements:
            self.walk(statement)

    def walk_Statement(self, node: ast.Statement):
        if node.op != "=":
            return

        primary_node = compiler.get_primary_ast_variable(node)
        primary_name = primary_node.name
        primary_variable = self.variable_table[primary_name]

        if node.left.name != primary_name:
            msg = f"Not Implemented Error: The left hand side of assignment must be the primary variable for now"
            raise AstException("computing transformed parameters", msg, node)

        self.traced_nodes = []

        self.walk(node.left)
        self.walk(node.right)

        arguments = tuple(self.trace_table[traced_node].array for traced_node in self.traced_nodes)

        def mapper(*mapper_arguments):
            executor = CodeExecutor(dict(zip(self.traced_nodes, mapper_arguments)), self.parameters)
            return executor.walk(node.right)

        # We can't be overwriting parameters
        assert primary_name not in self.parameters

        if primary_variable.argument_count > 0:
            self.parameters[primary_name] = jax.vmap(mapper)(*arguments)
        else:
            self.parameters[primary_name] = mapper(*arguments)

    def walk_Variable(self, node: ast.Variable):
        node_variable = self.variable_table[node.name]
        if node_variable.argument_count > 0:
            if node in self.trace_table:
                self.traced_nodes.append(node)


@dataclass
class EvaluateDensityWalker(walker.RatWalker):
    variable_table: VariableTable
    trace_table: TraceTable
    parameters: Dict[str, Any]
    traced_nodes: List[ast.Variable] = field(default_factory=list)

    def walk_Program(self, node: ast.Program):
        target = 0.0
        for statement in node.statements:
            target += self.walk(statement)
        return target

    def walk_Statement(self, node: ast.Statement):
        if node.op != "~":
            return 0.0

        primary_node = compiler.get_primary_ast_variable(node)
        primary_name = primary_node.name
        primary_variable = self.variable_table[primary_name]

        self.traced_nodes = []

        self.walk(node.left)
        self.walk(node.right)

        arguments = tuple(self.trace_table[traced_node].array for traced_node in self.traced_nodes)

        def mapper(*mapper_arguments):
            executor = CodeExecutor(dict(zip(self.traced_nodes, mapper_arguments)), self.parameters)
            return executor.walk(node)

        if primary_variable.argument_count > 0:
            return jax.numpy.sum(jax.vmap(mapper)(*arguments))
        else:
            return mapper(*arguments)

    def walk_Variable(self, node: ast.Variable):
        node_variable = self.variable_table[node.name]
        if node_variable.argument_count > 0:
            if node in self.trace_table:
                self.traced_nodes.append(node)


@dataclass
class ConstraintFinder(walker.RatWalker):
    variable_table: VariableTable
    trace_table: TraceTable
    constraints: Dict[str, Union[float, numpy.ndarray]] = field(default_factory=dict)

    def walk_Variable(self, node: ast.Variable):
        if node.constraints:
            assert node.name not in self.constraints

            variable = self.variable_table[node.name]

            node_constraints = {}

            if node.constraints.left:
                left_name = node.constraints.left.name
                values = self.trace_table[node.constraints.left.value].array
                node_constraints[left_name] = values if variable.argument_count > 0 else values[0]

            if node.constraints.right:
                right_name = node.constraints.right.name
                values = self.trace_table[node.constraints.right.value].array
                node_constraints[right_name] = values if variable.argument_count > 0 else values[0]

            self.constraints[node.name] = node_constraints


class Model:
    base_df_dict: Dict[str, pandas.DataFrame]
    program: ast.Program
    variable_table: VariableTable
    trace_table: TraceTable

    @property
    def size(self):
        return self.variable_table.unconstrained_parameter_size

    @partial(jax.jit, static_argnums=(0,))
    def constrain(self, unconstrained_parameter_vector: numpy.ndarray):
        constraint_finder = ConstraintFinder(self.variable_table, self.trace_table)
        constraint_finder.walk(self.program)
        jacobian_adjustments = 0.0
        parameters = {}
        used = 0

        for name in self.variable_table:
            record = self.variable_table[name]

            if not isinstance(record, SampledVariableRecord):
                continue

            # This assumes that unconstrained parameter indices for a parameter is allocated in a contiguous fashion.
            if len(record.subscripts) > 0:
                unconstrained = unconstrained_parameter_vector[used: used + len(record)]
                used += len(record)
            else:
                unconstrained = unconstrained_parameter_vector[used]
                used += 1

            if name in constraint_finder.constraints:
                variable_constraints = constraint_finder.constraints[name]

                if "lower" in variable_constraints and "upper" not in variable_constraints:
                    constrained, jacobian_adjustment = constraints.lower(unconstrained, variable_constraints["lower"])
                elif "lower" not in constraints and "upper" in variable_constraints:
                    constrained, jacobian_adjustment = constraints.upper(unconstrained, variable_constraints["upper"])
                else:  # "lower" in constraints and "upper" in constraints:
                    constrained, jacobian_adjustment = constraints.finite(
                        unconstrained, variable_constraints["lower"], variable_constraints["upper"]
                    )

                jacobian_adjustments += jax.numpy.sum(jacobian_adjustment)
            else:
                constrained = unconstrained

            parameters[name] = constrained

        return jacobian_adjustments, parameters

    @partial(jax.jit, static_argnums=(0,))
    def constrain_and_transform(self, unconstrained_parameter_vector: numpy.ndarray):
        jacobian, parameters = self.constrain(unconstrained_parameter_vector)

        transform = TransformedParametersFunctionGenerator(self.variable_table, self.trace_table, parameters)
        transform.walk(self.program)

        return jacobian, parameters

    @partial(jax.jit, static_argnums=(0, 2))
    def log_density(self, unconstrained_parameter_vector: numpy.ndarray, include_jacobian: bool = True):
        jacobian, parameters = self.constrain(unconstrained_parameter_vector)

        # Modify parameters in place
        transform = TransformedParametersFunctionGenerator(self.variable_table, self.trace_table, parameters)
        transform.walk(self.program)

        likelihood = EvaluateDensityWalker(self.variable_table, self.trace_table, parameters)
        target = likelihood.walk(self.program)

        return target + (jacobian if include_jacobian else 0.0)

    def log_density_no_jac(self, unconstrained_parameter_vector: numpy.ndarray):
        return self.log_density(unconstrained_parameter_vector, False)

    def _prepare_draws_and_dfs(self, device_unconstrained_draws):
        constrain_and_transform_jax = jax.jit(jax.vmap(jax.vmap(lambda x: self.constrain_and_transform(x))))

        _, constrained_draws = constrain_and_transform_jax(device_unconstrained_draws)

        # # Copy back to numpy arrays
        return {name: numpy.array(draws) for name, draws in constrained_draws.items()}, self.base_df_dict

    def __init__(self, model_string: str, data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]],
                 max_trace_iterations: int = 50):
        """
        Create a model from some data (`data`) and a model (specified as a string, `model_string`).

        If compile_path is not None, then write the compiled model to the given path (will only overwrite
        existing files if the overwrite flag is true)

        The parsed_lines argument is for creating a model from an intermediate representation -- likely
        deprecated soon.

        max_trace_iterations determines how many tracing iterations the program will do to resolve
        parameter domains
        """
        data_dict = {}

        match data:
            case pandas.DataFrame():
                data_dict["__default"] = data.reset_index().copy()
            case dict():
                for key, value in data.items():
                    data_dict[key] = value.reset_index().copy()
            case _:
                raise Exception("Data must either be pandas data frames or a dict of pandas dataframes")

        # Parse the model to get AST
        semantics = ModelBuilderSemantics()
        parser = RatParser(semantics=semantics)
        # TODO: This lambda is just here to make sure pylance formatting works -- should work
        # without it as well
        self.program = (lambda: parser.parse(model_string))()

        # Compile the model
        rat_compiler = RatCompiler(data_dict, self.program, max_trace_iterations=max_trace_iterations)
        self.variable_table = rat_compiler.variable_table
        self.trace_table = rat_compiler.trace_table

        self.base_df_dict = {}

        for variable_name in self.variable_table:
            record = self.variable_table[variable_name]
            # The base_dfs are used for putting together the output
            if isinstance(record, DynamicVariableRecord):
                rows = list(record.itertuples())
                self.base_df_dict[variable_name] = pandas.DataFrame.from_records(rows, columns=record.subscripts)

    @staticmethod
    def from_file(
        filename: str,
        data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]],
        max_trace_iterations: int = 50,
    ):
        with open(filename) as f:
            return Model(f.read(), data, max_trace_iterations)
