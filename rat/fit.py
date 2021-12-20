import arviz
import blackjax
import blackjax.nuts
import blackjax.inference
import numpy
import pandas
from typing import List, Dict

from . import constraints
from . import model


class Fit:
    constrained_variables: Dict[str, numpy.array]
    base_dfs: Dict[str, pandas.DataFrame]
    draw_dfs: Dict[str, pandas.DataFrame]

    def build_constrained_dfs(
        self, constrained_variables: Dict[str, numpy.array], base_dfs: Dict[str, pandas.DataFrame]
    ) -> Dict[str, pandas.DataFrame]:
        """
        Re-sort parameters stored as dictionary of arrays into dataframes

        Scalar parameter arrays are expected to have shape (num_draws, num_chains) or (1, num_draws, num_chains)
        Vector parameter arrays are expected to have shape (size, num_draws, num_chains)
        """
        draw_dfs: Dict[str, pandas.DataFrame] = {}
        for name in constrained_variables:
            constrained_variable = constrained_variables[name]
            if len(constrained_variable.shape) == 2:
                size = 1
            else:
                size = constrained_variable.shape[-1]
            num_draws = constrained_variable.shape[0]
            num_chains = constrained_variable.shape[1]

            base_df = base_dfs[name]

            df = pandas.concat([base_df] * num_draws * num_chains, ignore_index=True)
            df["chain"] = numpy.repeat(numpy.arange(num_chains), size * num_draws)
            df["draw"] = numpy.tile(numpy.repeat(numpy.arange(num_draws), size), num_chains)
            df["value"] = constrained_variable.flatten(order="C")
            draw_dfs[name] = df

        return draw_dfs

    def draws(self, parameter_name: str) -> pandas.DataFrame:
        return self.draw_dfs[parameter_name]


class OptimizationFit(Fit):
    def check_convergence_and_select_one_chain(self, draw_dfs, tolerance):
        # Check that all the optimization solutions are vaguely close to each other
        for name, draw_df in draw_dfs.items():
            group_columns = list(set(draw_df.columns) - set(["value", "chain"]))
            if len(group_columns) > 0:
                grouped_df = draw_df.groupby(group_columns)
                median_df = grouped_df.agg({"value": numpy.median}).rename(columns={"value": "median"})
                converged_df = (
                    draw_df.merge(median_df, on=group_columns, how="left")
                    .assign(absolute_difference=lambda df: (df["value"] - df["median"]).abs())
                    .assign(tolerance=lambda df: numpy.maximum(tolerance, tolerance * df["value"].abs()))
                    .assign(converged=lambda df: df["absolute_difference"] < df["tolerance"])
                )
                not_converged = ~converged_df["converged"]
                if sum(not_converged) > 0:
                    row = converged_df[not_converged].iloc[0]
                    raise Exception(f"Difference optimizations [] didn't converge to within tolerance")
            else:
                values = draw_df["value"]
                median = numpy.median(values)
                absolute_differences = numpy.abs(draw_df["value"] - median)
                thresholds = numpy.maximum(tolerance, tolerance * numpy.abs(values))
                if any(absolute_differences > thresholds):
                    values_string = ",".join(str(value) for value in values)
                    differences_string = ",".join(str(difference) for difference in absolute_differences)
                    thresholds_string = ",".join(str(threshold) for threshold in thresholds)
                    raise Exception(
                        f"For {name} [{values_string}], absolute differences [{differences_string}] exceed thresholds [{thresholds_string}]"
                    )

        # Only copy one optimization result
        output_draw_dfs = {}
        for name, draw_df in draw_dfs.items():
            output_draw_dfs[name] = draw_df[draw_df["chain"] == 0]

        return output_draw_dfs

    def __init__(self, constrained_variables: Dict[str, numpy.array], base_dfs: Dict[str, pandas.DataFrame], tolerance):
        self.constrained_variables = constrained_variables
        self.base_dfs = base_dfs
        draw_dfs = self.build_constrained_dfs(constrained_variables, base_dfs)
        self.draw_dfs = self.check_convergence_and_select_one_chain(draw_dfs, tolerance)


class SampleFit(Fit):
    diag_dfs: pandas.DataFrame

    def __init__(self, constrained_variables: Dict[str, numpy.array], base_dfs: Dict[str, pandas.DataFrame]):
        self.constrained_variables = constrained_variables
        self.base_dfs = base_dfs

        # Unpack draws into dataframes
        self.draw_dfs = self.build_constrained_dfs(constrained_variables, base_dfs)

        self.diag_dfs = {}
        for name, constrained_variable in constrained_variables.items():
            # Compute ess/rhat, must reshape from (draw, chain, param) to (chain, draw, param)
            if len(constrained_variable.shape) == 3:
                arviz_constrained_variable = numpy.swapaxes(constrained_variable, 0, 1)
            else:
                arviz_constrained_variable = numpy.expand_dims(constrained_variable.transpose(), 2)
            x_draws = arviz.convert_to_dataset(arviz_constrained_variable)
            ess = arviz.ess(x_draws)["x"].to_numpy()
            rhat = arviz.rhat(x_draws)["x"].to_numpy()

            self.diag_dfs[name] = base_dfs[name].assign(ess=ess, rhat=rhat)

    def diag(self, parameter_name: str) -> pandas.DataFrame:
        return self.diag_dfs[parameter_name]
