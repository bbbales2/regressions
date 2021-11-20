import blackjax
import blackjax.nuts
import blackjax.inference
import numpy
import pandas
from typing import List, Dict


class Fit:
    draw_dfs: Dict[str, pandas.DataFrame]

    def __init__(self, draw_dfs: Dict[str, pandas.DataFrame]):
        self.draw_dfs = draw_dfs

    def draws(self, parameter_name: str) -> pandas.DataFrame:
        return self.draw_dfs[parameter_name]


class OptimizationFit(Fit):
    def __init__(self, draw_dfs: Dict[str, pandas.DataFrame], tolerance=1e-2):
        self.draw_dfs = {}

        # Check that all the optimization solutions are vaguely close to each other
        for name, draw_df in draw_dfs.items():
            group_columns = list(set(draw_df.columns) - set(["value", "draw"]))
            if len(group_columns) > 0:
                grouped_df = draw_df.groupby(group_columns)
                median_df = grouped_df.agg({"value": numpy.median}).rename(
                    columns={"value": "median"}
                )
                converged_df = (
                    draw_df.merge(median_df, on=group_columns, how="left")
                    .assign(
                        absolute_difference=lambda df: (
                            df["value"] - df["median"]
                        ).abs()
                    )
                    .assign(
                        tolerance=lambda df: numpy.maximum(
                            tolerance, tolerance * df["value"].abs()
                        )
                    )
                    .assign(
                        converged=lambda df: df["absolute_difference"] < df["tolerance"]
                    )
                )
                not_converged = ~converged_df["converged"]
                if sum(not_converged) > 0:
                    row = converged_df[not_converged].iloc[0]
                    raise Exception(
                        f"Difference optimizations [] didn't converge to within tolerance"
                    )
            else:
                values = draw_df["value"]
                median = numpy.median(values)
                absolute_differences = numpy.abs(draw_df["value"] - median)
                thresholds = numpy.maximum(tolerance, tolerance * numpy.abs(values))
                if any(absolute_differences > thresholds):
                    values_string = ",".join(str(value) for value in values)
                    differences_string = ",".join(
                        str(difference) for difference in absolute_differences
                    )
                    thresholds_string = ",".join(
                        str(threshold) for threshold in thresholds
                    )
                    raise Exception(
                        f"For {name} [{values_string}], absolute differences [{differences_string}] exceed thresholds [{thresholds_string}]"
                    )

        # Only copy one optimization result
        for name, draw_df in draw_dfs.items():
            self.draw_dfs[name] = draw_df[draw_df["draw"] == 0]
