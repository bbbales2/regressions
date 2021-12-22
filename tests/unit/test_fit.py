import numpy
import pandas
import pytest
import tempfile
from rat import fit, model


@pytest.fixture
def scalar_base_dfs():
    return {"x": pandas.DataFrame()}


@pytest.fixture
def scalar_constrained_draws():
    constrained_draws = {"x": numpy.array([10.1, 10.08, 10.05, 10.03]).reshape((1, 4, 1))}
    return constrained_draws


@pytest.fixture
def vector_base_dfs():
    data_df = pandas.DataFrame({"group": [0, 1]})
    return {"x": data_df}


@pytest.fixture
def vector_constrained_draws():
    constrained_draws = numpy.array([20.2, 10.1, 20.17, 10.08, 20.1, 10.05, 20.13, 10.03]).reshape((4, 1, 2))
    return {"x": constrained_draws}


@pytest.fixture
def optimization_fit(scalar_constrained_draws, scalar_base_dfs):
    return fit.OptimizationFit._from_constrained_variables(scalar_constrained_draws, scalar_base_dfs, tolerance=1e-2)


@pytest.fixture
def sample_fit(scalar_constrained_draws, scalar_base_dfs):
    return fit.SampleFit._from_constrained_variables(scalar_constrained_draws, scalar_base_dfs)


def test_optimization_save_and_reload(optimization_fit):
    with tempfile.TemporaryDirectory() as folder:
        optimization_fit.save(folder)

        with pytest.raises(FileExistsError):
            optimization_fit.save(folder)

        optimization_fit.save(folder, overwrite=True)

        copy_fit = fit.load(folder)

        for name in optimization_fit.draw_dfs:
            original = optimization_fit.draws(name)
            copy = copy_fit.draws(name)

            assert len(original) == len(copy)

            inner = copy.merge(original, on=list(copy.columns), how="inner")

            assert len(inner) == len(copy)


def test_sample_save_and_reload(sample_fit):
    with tempfile.TemporaryDirectory() as folder:
        sample_fit.save(folder)

        with pytest.raises(FileExistsError):
            sample_fit.save(folder)

        sample_fit.save(folder, overwrite=True)

        copy_fit = fit.load(folder)

        for name in sample_fit.draw_dfs:
            original = sample_fit.draws(name)
            copy = copy_fit.draws(name)

            assert len(original) == len(copy)

            inner = copy.merge(original, on=list(copy.columns), how="inner")

            assert len(inner) == len(copy)

        for name in sample_fit.diag_dfs:
            original = sample_fit.diag(name)
            copy = copy_fit.diag(name)

            assert len(original) == len(copy)

            inner = copy.merge(original, on=list(copy.columns), how="inner")

            assert len(inner) == len(copy)


def test_optimization_fit(optimization_fit):
    # Just make sure optimization_fit gets created
    pass


def test_optimization_fit_error(scalar_constrained_draws, scalar_base_dfs):
    # Throws an error because the given results are not within tolerance of each other
    with pytest.raises(Exception):
        fit.OptimizationFit._from_constrained_variables(scalar_constrained_draws, scalar_base_dfs, tolerance=1e-3)


def test_optimization_fit_vector(vector_constrained_draws, vector_base_dfs):
    fit.OptimizationFit._from_constrained_variables(vector_constrained_draws, vector_base_dfs, tolerance=1e-2)


def test_optimization_fit_error_vector(vector_constrained_draws, vector_base_dfs):
    # Throws an error because the given results are not within tolerance of each other
    with pytest.raises(Exception):
        fit.OptimizationFit._from_constrained_variables(vector_constrained_draws, vector_base_dfs_draws, tolerance=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
