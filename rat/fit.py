import arviz
import blackjax
import blackjax.nuts
import blackjax.inference
import glob
import numpy
import os
import pandas
from typing import List, Dict, Union

from . import constraints
from . import model


def _check_writeable(path, overwrite):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Write failed; {path} exists and overwrite is not True")

def _write_dict_into_folder(dfs : Dict[str, pandas.DataFrame], folder : str, overwrite):
    for name, df in dfs.items():
        df_path = os.path.join(folder, name + ".parquet")
        _check_writeable(df_path, overwrite)
        df.to_parquet(df_path)

def _read_folder_into_dict(folder : str):
    dfs = {}
    for path in glob.glob(f"{folder}/*.parquet"):
        name, ext = os.path.splitext(os.path.basename(path))
        dfs[name] = pandas.read_parquet(path)
    return dfs

def _build_constrained_dfs(
    constrained_variables: Dict[str, numpy.array], base_dfs: Dict[str, pandas.DataFrame]
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

def _check_convergence_and_select_one_chain(draw_dfs, tolerance):
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


class Fit:
    """
    Parent class for optimization/MCMC results
    """
    draw_dfs: Dict[str, pandas.DataFrame]

    def draws(self, parameter_name: str) -> pandas.DataFrame:
        """
        Get the draws for a given parameter with columns for the
        subscripts. For optimization results there is only one result
        (the optimum) and no chains column.
        """
        return self.draw_dfs[parameter_name]
    
    def save(self, folder, overwrite = False):
        """
        Save results to a folder. If overwrite is true, overwrite
        existing files and folders
        """
        draws_folder = os.path.join(folder, "draws")
        if os.path.exists(draws_folder) and overwrite is not True:
            raise FileExistsError(f"Folder {folder} already exists and overwrite is not True")
        _write_dict_into_folder(self.draw_dfs, draws_folder, overwrite)
        
        type_path = os.path.join(folder, "type")
        _check_writeable(type_path, overwrite = overwrite)
        with open(type_path, "w") as f:
            f.write(self.__class__.__name__)

class OptimizationFit(Fit):
    """
    Stores optimization results
    """
    def __init__(self, draw_dfs):
        self.draw_dfs = draw_dfs

    @classmethod
    def _from_constrained_variables(cls, constrained_variables: Dict[str, numpy.array], base_dfs: Dict[str, pandas.DataFrame], tolerance):
        draw_dfs = _build_constrained_dfs(constrained_variables, base_dfs)
        return cls(_check_convergence_and_select_one_chain(draw_dfs, tolerance))

class SampleFit(Fit):
    """
    Stores draws from an MCMC calculation
    """
    diag_dfs: Dict[str, pandas.DataFrame]

    def __init__(self, draw_dfs, diag_dfs):
        self.draw_dfs = draw_dfs
        self.diag_dfs = diag_dfs
    
    @classmethod
    def _from_constrained_variables(cls, constrained_variables: Dict[str, numpy.array], base_dfs: Dict[str, pandas.DataFrame]):
        # Unpack draws into dataframes
        draw_dfs = _build_constrained_dfs(constrained_variables, base_dfs)
        diag_dfs = {}
        for name, constrained_variable in constrained_variables.items():
            # Compute ess/rhat, must reshape from (draw, chain, param) to (chain, draw, param)
            if len(constrained_variable.shape) == 3:
                arviz_constrained_variable = numpy.swapaxes(constrained_variable, 0, 1)
            else:
                arviz_constrained_variable = numpy.expand_dims(constrained_variable.transpose(), 2)
            x_draws = arviz.convert_to_dataset(arviz_constrained_variable)
            ess = arviz.ess(x_draws)["x"].to_numpy()
            rhat = arviz.rhat(x_draws)["x"].to_numpy()

            diag_dfs[name] = base_dfs[name].assign(ess=ess, rhat=rhat)

        return cls(draw_dfs, diag_dfs)

    def diag(self, parameter_name: str) -> pandas.DataFrame:
        """
        Get diagnostic dataframe for a given parameter. Diagnostics are currently
        effective sample size and rhat
        """
        return self.diag_dfs[parameter_name]
    
    def save(self, folder, overwrite = False):
        """
        Save the SampleFit object to a folder. If overwrite is true, then overwrite
        existing files and use existing folders
        """
        super(SampleFit, self).save(folder, overwrite = overwrite)

        sample_folder = os.path.join(folder, "SampleFit")
        _write_dict_into_folder(self.diag_dfs, sample_folder, overwrite)

def load(folder : str) -> Union[OptimizationFit, SampleFit]:
    """
    Load an OptimizationFit/SampleFit from the folder in which
    it was saved
    """
    type_path = os.path.join(folder, "type")

    with open(type_path) as f:
        type_string = f.readline().strip()
    
    draws_folder = os.path.join(folder, "draws")
    draw_dfs = _read_folder_into_dict(draws_folder)

    if type_string == "OptimizationFit":
        return OptimizationFit(draw_dfs)
    elif type_string == "SampleFit":
        sample_folder = os.path.join(folder, "SampleFit")
        diag_dfs = _read_folder_into_dict(sample_folder)
        return SampleFit(draw_dfs, diag_dfs)
    else:
        raise TypeError(f"Unrecognized type {type_string} found in {type_path} when loading {folder}")