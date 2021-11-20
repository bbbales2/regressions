import pandas
import pytest
from rat import fit


@pytest.fixture
def scalar_draws_df():
    value = [10.1, 10.08, 10.05, 10.03]
    draw = [1, 2, 3, 4]
    df = pandas.DataFrame({"value": value, "draw": draw})
    return {"var": df}


@pytest.fixture
def vector_draws_df():
    value = [20.2, 20.17, 20.1, 20.13, 10.1, 10.08, 10.05, 10.03]
    group = [1, 1, 1, 1, 4, 4, 4, 4]
    draw = [1, 2, 3, 4, 1, 2, 3, 4]
    df = pandas.DataFrame({"value": value, "group": group, "draw": draw})
    return {"var": df}


def test_optimization_fit(scalar_draws_df):
    fit.OptimizationFit(scalar_draws_df)


def test_optimization_fit_error(scalar_draws_df):
    with pytest.raises(Exception):
        fit.OptimizationFit(scalar_draws_df, tolerance=1e-3)


def test_optimization_fit_vector(vector_draws_df):
    fit.OptimizationFit(vector_draws_df)


def test_optimization_fit_error_vector(vector_draws_df):
    with pytest.raises(Exception):
        fit.OptimizationFit(vector_draws_df, tolerance=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
