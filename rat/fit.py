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
    model: model.Model
    draw_dfs: Dict[str, pandas.DataFrame]

    def constrain_draws(self, unconstrained_draws: numpy.array) -> numpy.array:
        """
        Map input array of size (num_chains, num_draws, num_parameters) to
        output with the same shape, but constrain the packed parameters in the
        last dimension
        """
        draws = numpy.zeros(unconstrained_draws.shape)

        for name, offset, size in zip(self.model.parameter_names, self.model.parameter_offsets, self.model.parameter_sizes):
            filled_size = size if size is not None else 1

            value = unconstrained_draws[:, :, offset : offset + filled_size]

            variable = self.model.parameter_variables[name]
            lower = variable.lower
            upper = variable.upper
            if lower > float("-inf") and upper == float("inf"):
                value, _ = constraints.lower(value, lower)
            elif lower == float("inf") and upper < float("inf"):
                value, _ = constraints.upper(value, upper)
            elif lower > float("inf") and upper < float("inf"):
                value, _ = constraints.finite(value, lower, upper)

            draws[:, :, offset : offset + filled_size] = value

        return draws

    def build_draw_dfs(self, draws: numpy.array) -> Dict[str, pandas.DataFrame]:
        """
        Unpack parameters from draws stored in array of shape
        (num_chains, num_draws, num_parameters) into long-format
        dataframes indexed by parameter name
        """
        draw_dfs: Dict[str, pandas.DataFrame] = {}
        num_chains = draws.shape[0]
        num_draws = draws.shape[1]
        draw_series = list(range(num_draws))
        for name, offset, size in zip(self.model.parameter_names, self.model.parameter_offsets, self.model.parameter_sizes):
            filled_size = size if size is not None else 1

            dfs = []
            for chain in range(num_chains):
                if size is not None:
                    df = self.model.parameter_variables[name].index.base_df.copy()
                    for draw in range(num_draws):
                        df["value"] = draws[chain, draw, offset : offset + size]
                        if num_chains > 1:
                            df["chain"] = chain
                        if num_draws > 1:
                            df["draw"] = draw
                        dfs.append(df)
                else:
                    series = draws[chain, :, offset]
                    df = pandas.DataFrame({"value": series})
                    if num_chains > 1:
                        df["chain"] = chain
                    if num_draws > 1:
                        df["draw"] = draw_series
                    dfs.append(df)

            draw_dfs[name] = pandas.concat(dfs, ignore_index=True)
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

    def __init__(self, model: model.Model, unconstrained_draws: numpy.array, tolerance):
        self.model = model
        draws = self.constrain_draws(unconstrained_draws)
        draw_dfs = self.build_draw_dfs(draws)
        self.draw_dfs = self.check_convergence_and_select_one_chain(draw_dfs, tolerance)


class SampleFit(Fit):
    ess_dfs: pandas.DataFrame
    rhat_dfs: pandas.DataFrame

    def __init__(self, model: model.Model, unconstrained_draws: numpy.array):
        self.model = model
        # Constrain the draws
        draws = self.constrain_draws(unconstrained_draws)

        # Compute ess/rhat
        x_draws = arviz.convert_to_dataset(draws)
        ess = arviz.ess(x_draws)["x"].to_numpy().reshape((1, 1, -1))
        rhat = arviz.rhat(x_draws)["x"].to_numpy().reshape((1, 1, -1))

        self.ess_dfs = self.build_draw_dfs(ess)
        self.rhat_dfs = self.build_draw_dfs(rhat)

        # Unpack draws into dataframes
        self.draw_dfs = self.build_draw_dfs(draws)

    def ess(self, parameter_name: str) -> pandas.DataFrame:
        return self.ess_dfs[parameter_name]

    def rhat(self, parameter_name: str) -> pandas.DataFrame:
        return self.rhat_dfs[parameter_name]
