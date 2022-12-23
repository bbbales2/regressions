from functools import partial
import jax
import jax.experimental.host_callback
import numpy
import pandas
import tatsu
from typing import Dict, Union, List
from tatsu.model import ModelBuilderSemantics

from rat.compiler import StatementComponent
from rat.parser import RatParser
from rat.trace_table import TraceTable
from rat.variable_table import VariableTable, SampledVariableRecord, DynamicVariableRecord
from rat import ast
from rat.walker import flatten_ast


class Model:
    base_df_dict: Dict[str, pandas.DataFrame]
    program: ast.Program
    variable_table: VariableTable
    trace_table: TraceTable
    statement_components: List[StatementComponent]

    @property
    def size(self):
        return self.variable_table.unconstrained_parameter_size

    @partial(jax.jit, static_argnums=(0,))
    def temporary_parameters(self, unconstrained_parameter_vector: numpy.ndarray):
        parameters = {}
        used = 0

        for name in self.variable_table:
            record = self.variable_table[name]

            if not isinstance(record, SampledVariableRecord):
                continue

            # This assumes that unconstrained parameter indices for a parameter is allocated in a contiguous fashion.
            if len(record.subscripts) > 0:
                unconstrained = unconstrained_parameter_vector[used : used + len(record)]
                used += len(record)
            else:
                unconstrained = unconstrained_parameter_vector[used]
                used += 1

            parameters[name] = unconstrained

        return parameters

    @partial(jax.jit, static_argnums=(0, 2))
    def compute_parameters(self, x: numpy.ndarray):
        # log_jacobian = self.variable_table.transform(x)
        parameters = self.temporary_parameters(x)

        for statement_component in reversed(self.statement_components):
            target_increment, parameters = statement_component.evaluate(parameters, False)

        return parameters

    @partial(jax.jit, static_argnums=(0, 2))
    def log_density(self, x: numpy.ndarray, include_jacobian=True):
        # log_jacobian = self.variable_table.transform(x)
        parameters = self.temporary_parameters(x)

        target = 0.0
        for statement_component in reversed(self.statement_components):
            target_increment, parameters = statement_component.evaluate(parameters, include_jacobian)
            target += jax.numpy.sum(target_increment)

        return target

    def log_density_no_jac(self, unconstrained_parameter_vector: numpy.ndarray):
        return self.log_density(unconstrained_parameter_vector, False)

    def _prepare_draws_and_dfs(self, device_unconstrained_draws):
        compute_parameters_jax = jax.jit(jax.vmap(jax.vmap(self.compute_parameters)))

        constrained_draws = compute_parameters_jax(device_unconstrained_draws)

        # # Copy back to numpy arrays
        return {name: numpy.array(draws) for name, draws in constrained_draws.items()}, self.base_df_dict

    def __init__(self, model_string: str, data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]], max_trace_iterations: int = 50):
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
        semantics = ModelBuilderSemantics(types=[])
        parser = RatParser(semantics=semantics)
        # TODO: This lambda is just here to make sure pylance formatting works -- should work
        # without it as well
        program = (lambda: parser.parse(model_string))()

        # Compile the model
        variable_table = VariableTable()
        statement_components = [
            StatementComponent(node, variable_table, data_dict) for node in flatten_ast(program) if isinstance(node, tatsu.synth.Statement)
        ]

        self.program = program
        self.variable_table = variable_table
        self.statement_components = statement_components

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
