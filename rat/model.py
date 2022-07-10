from concurrent.futures import ThreadPoolExecutor
import functools
import importlib.util
import jax
import jax.experimental.host_callback
import numpy
import os
import pandas
import scipy.optimize
import tempfile
import types
from typing import Callable, Dict, Union
from tatsu.model import ModelBuilderSemantics

from rat import subscript_table

from .compiler2 import RatCompiler
from .parser import RatParser
from .variable_table import VariableTable, VariableType
from . import fit
from . import nuts


class Model:
    log_density_jax: Callable[[numpy.array], float]
    log_density_jax_no_jac: Callable[[numpy.array], float]
    size: int
    base_df_dict: Dict[str, pandas.DataFrame]
    device_data: Dict[str, jax.numpy.array]
    device_subscript_indices: Dict[str, jax.numpy.array]
    device_first_in_group_indicators: Dict[str, jax.numpy.array]
    compiled_model: types.ModuleType

    def _constrain_and_transform_parameters(self, unconstrained_parameter_vector, pad=True):
        jacobian_adjustment, parameters = self.compiled_model.constrain_parameters(unconstrained_parameter_vector, pad)
        return jacobian_adjustment, self.compiled_model.transform_parameters(
            self.device_data, self.device_subscript_indices, self.device_first_in_group_indicators, parameters
        )

    def _log_density(self, include_jacobian, unconstrained_parameter_vector):
        # Evaluate model log density given model, data, subscripts and unconstrained parameters
        jacobian_adjustment, parameters = self._constrain_and_transform_parameters(unconstrained_parameter_vector, pad=True)
        target = self.compiled_model.evaluate_densities(self.device_data, self.device_subscript_indices, parameters)
        return target + (jacobian_adjustment if include_jacobian else 0.0)

    def _prepare_draws_and_dfs(self, device_unconstrained_draws):
        unconstrained_draws = numpy.array(device_unconstrained_draws)
        num_draws = unconstrained_draws.shape[0]
        num_chains = unconstrained_draws.shape[1]
        constrained_draws = {}

        constrain_and_transform_parameters_no_pad_jax = jax.jit(lambda x: self._constrain_and_transform_parameters(x, pad=False))
        for draw in range(num_draws):
            for chain in range(num_chains):
                jacobian_adjustment, device_constrained_variables = constrain_and_transform_parameters_no_pad_jax(
                    unconstrained_draws[draw, chain]
                )

                for name, device_constrained_variable in device_constrained_variables.items():
                    if name not in constrained_draws:
                        constrained_draws[name] = numpy.zeros((num_draws, num_chains) + device_constrained_variable.shape)
                    constrained_draws[name][draw, chain] = numpy.array(device_constrained_variable)

        # Copy back to numpy arrays
        return constrained_draws, self.base_df_dict

    def __init__(
        self,
        data: Union[pandas.DataFrame, Dict[str, pandas.DataFrame]],
        model_string: str,
        max_trace_iterations: int = 50,
        compile_path: str = None,
        overwrite: bool = False,
    ):
        """
        Create a model from some data (`data`) and a model (specified as a string, `model_string`).

        If compile_path is not None, then write the compiled model to the given path (will only overwrite
        existing files if the overwrite flag is true)

        The parsed_lines argument is for creating a model from an intermediate representation -- likely
        deprecated soon.

        max_trace_iterations determines how many tracing iterations the program will do to resolve
        parameter domains
        """
        match data:
            case pandas.DataFrame():
                data_names = data.columns
            case dict():
                data_names = set()
                for key, value in data.items():
                    if not isinstance(key, str):
                        raise Exception(f"Keys of dictionary form of data must be of type `str`, found {type(key)}")

                    match value:
                        case pandas.DataFrame():
                            for column in value.columns:
                                data_names.add(column)
                        case _:
                            raise Exception(f"Values of dictionary form of data must be pandas DataFrames, found {type(value)}")
            case _:
                raise Exception("Data must be a pandas DataFrame or a dictionary")

        # Parse the model to get AST
        semantics = ModelBuilderSemantics()
        parser = RatParser(semantics=semantics)
        # TODO: This lambda is just here to make sure pylance formatting works -- should work
        # without it as well
        program_ast = (lambda: parser.parse(model_string))()

        # Compile the model
        compiler = RatCompiler(data, program_ast, model_string, max_trace_iterations=max_trace_iterations)
        model_source_string, variable_table, subscript_table = compiler.compile()

        # Write model source to file and compile and import it
        if compile_path is None:
            self.working_dir = tempfile.TemporaryDirectory(prefix="rat.")

            model_source_file = os.path.join(self.working_dir.name, "model_source.py")
        else:
            self.working_dir = os.path.dirname(compile_path)
            if self.working_dir != "":
                os.makedirs(self.working_dir, exist_ok=True)

            if os.path.exists(compile_path) and not overwrite:
                raise FileExistsError(f"Compile path {compile_path} already exists and will not be overwritten")

            model_source_file = compile_path

        with open(model_source_file, "w") as f:
            f.write(model_source_string)

        spec = importlib.util.spec_from_file_location("compiled_model", model_source_file)
        compiled_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compiled_model)
        self.compiled_model = compiled_model

        self.base_df_dict = {}
        self.device_data = {}
        self.device_subscript_indices = {}

        for variable_name in variable_table:
            record = variable_table[variable_name]
            # The base_dfs are used for putting together the output
            if record.variable_type != VariableType.DATA:
                if record.base_df is not None:
                    self.base_df_dict[variable_name] = record.base_df
                else:
                    self.base_df_dict[variable_name] = pandas.DataFrame()

            # Copy data to jax device
            for name, data in record.get_numpy_arrays():
                self.device_data[name] = jax.device_put(data)

        for record in subscript_table.values():
            # Copy subscript indices to jax device
            self.device_subscript_indices[record.name] = jax.device_put(record.array)

        # Copy first in group indicators to jax device
        # TODO: Remove?
        self.device_first_in_group_indicators = {}

        self.log_density_jax = jax.jit(functools.partial(self._log_density, True))
        self.log_density_jax_no_jac = jax.jit(functools.partial(self._log_density, False))
        self.size = self.compiled_model.unconstrained_parameter_size

    def optimize(self, init=2, chains=4, retries=5, tolerance=1e-2):
        """
        Maximize the log density. `chains` difference optimizations are initialized.

        An error is thrown if the different solutions are not all within tolerance of the
        median solution for each parameter. If only one chain is used, the tolerance is
        ignored.

        If any optimization fails, retry up to `retries` number of times.

        Initialize parameters in unconstrained space uniformly [-2, 2].
        """

        def negative_log_density(x):
            return -self.log_density_jax_no_jac(x.astype(numpy.float32))

        grad = jax.jit(jax.grad(negative_log_density))

        def grad_double(x):
            grad_device_array = grad(x)
            return numpy.array(grad_device_array).astype(numpy.float64)

        unconstrained_draws = numpy.zeros((1, chains, self.size))
        for chain in range(chains):
            for retry in range(retries):
                params = 2 * init * numpy.random.uniform(size=self.size) - init

                solution = scipy.optimize.minimize(negative_log_density, params, jac=grad_double, method="L-BFGS-B", tol=1e-9)

                if solution.success:
                    unconstrained_draws[0, chain] = solution.x
                    break
            else:
                raise Exception(f"Optimization failed on chain {chain} with message: {solution.message}")

        constrained_draws, base_dfs = self._prepare_draws_and_dfs(unconstrained_draws)
        return fit.OptimizationFit._from_constrained_variables(constrained_draws, base_dfs, tolerance=tolerance)

    def sample(self, num_draws=200, num_warmup=1000, chains=4, init=2, thin=1, target_acceptance_rate=0.85):
        """
        Sample the target log density using NUTS.

        Sample using `chains` different chains with parameters initialized in unconstrained
        space [-2, 2]. Use `num_warmup` draws to warmup and collect `num_draws` draws in each
        chain after warmup.

        If `thin` is greater than 1, then compute internally `num_draws * thin` draws and
        output only every `thin` draws (so the output is size `num_draws`).

        `target_acceptance_rate` is the target acceptance rate for adaptation. Should be less
        than one and greater than zero.
        """
        # Currently only doing warmup on one chain
        initial_position = 2 * init * numpy.random.uniform(size=(self.size)) - init

        assert target_acceptance_rate < 1.0 and target_acceptance_rate > 0.0
        assert num_warmup > 200

        def negative_log_density(q):
            return -self.log_density_jax(q)

        potential = nuts.Potential(negative_log_density, chains, self.size)
        rng = numpy.random.default_rng()

        # Ordered as (draws, chains, param)
        unconstrained_draws = numpy.zeros((num_draws, chains, self.size))
        leapfrog_steps = numpy.zeros((num_draws, chains), dtype=int)
        divergences = numpy.zeros((num_draws, chains), dtype=bool)

        def generate_draws():
            stage_1_size = 100
            stage_3_size = 50
            stage_2_size = num_warmup - stage_1_size - stage_3_size

            initial_draw, stepsize, diagonal_inverse_metric = nuts.warmup(
                potential,
                rng,
                initial_position,
                target_accept_stat=target_acceptance_rate,
                stage_1_size=stage_1_size,
                stage_2_size=stage_2_size,
                stage_3_size=stage_3_size,
            )

            return nuts.sample(potential, rng, initial_draw, stepsize, diagonal_inverse_metric, num_draws, thin)

        with ThreadPoolExecutor(max_workers=chains) as e:
            results = []
            for chain in range(chains):
                results.append(e.submit(generate_draws))

            for chain, result in enumerate(results):
                unconstrained_draws[:, chain, :], leapfrog_steps[:, chain], divergences[:, chain] = result.result()

        constrained_draws, base_dfs = self._prepare_draws_and_dfs(unconstrained_draws)
        computational_diagnostic_variables = {"__leapfrog_steps": leapfrog_steps, "__divergences": divergences}

        for name, values in computational_diagnostic_variables.items():
            if name in constrained_draws:
                print(f"{name} already exists in sampler output, not writing diagnostic variable")
            else:
                constrained_draws[name] = values
                base_dfs[name] = pandas.DataFrame()

        return fit.SampleFit._from_constrained_variables(constrained_draws, base_dfs, computational_diagnostic_variables.keys())
