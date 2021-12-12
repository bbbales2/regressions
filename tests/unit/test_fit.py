import numpy
import pandas
import pytest
from rat import fit, model


@pytest.fixture
def scalar_base_dfs():
    return {"x": pandas.DataFrame()}


@pytest.fixture
def scalar_constrained_draws():
    constrained_draws = {"x": numpy.array([10.1, 10.08, 10.05, 10.03]).reshape((1, 1, 4))}
    return constrained_draws


@pytest.fixture
def vector_base_dfs():
    data_df = pandas.DataFrame({"group": [0, 1]})
    return {"x": data_df}


@pytest.fixture
def vector_constrained_draws():
    constrained_draws = numpy.array([20.2, 20.17, 20.1, 20.13, 10.1, 10.08, 10.05, 10.03]).reshape((2, 1, 4))
    return {"x": constrained_draws}


def test_optimization_fit(scalar_constrained_draws, scalar_base_dfs):
    fit.OptimizationFit(scalar_constrained_draws, scalar_base_dfs, tolerance=1e-2)


def test_optimization_fit_error(scalar_constrained_draws, scalar_base_dfs):
    # Throws an error because the given results are not within tolerance of each other
    with pytest.raises(Exception):
        fit.OptimizationFit(scalar_constrained_draws, scalar_base_dfs, tolerance=1e-3)


def test_optimization_fit_vector(vector_constrained_draws, vector_base_dfs):
    fit.OptimizationFit(vector_constrained_draws, vector_base_dfs, tolerance=1e-2)


def test_optimization_fit_error_vector(vector_constrained_draws, vector_base_dfs):
    # Throws an error because the given results are not within tolerance of each other
    with pytest.raises(Exception):
        fit.OptimizationFit(vector_constrained_draws, vector_base_dfs_draws, tolerance=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
