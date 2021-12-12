import numpy
import pandas
import pytest
from rat import fit, model


@pytest.fixture
def scalar_model():
    data_df = pandas.DataFrame({"y": [0.0]})

    model_string = """
    y ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    return model.Model(data_df, model_string=model_string)


@pytest.fixture
def scalar_unconstrained_draws():
    unconstrained_draws = numpy.array([10.1, 10.08, 10.05, 10.03]).reshape((4, 1, 1))
    return unconstrained_draws


@pytest.fixture
def vector_model():
    data_df = pandas.DataFrame({"y": [0.0, 1.0], "group": [0, 1]})

    model_string = """
    y ~ normal(mu[group], 1.5);
    mu[group] ~ normal(-0.5, 0.3);
    """

    return model.Model(data_df, model_string=model_string)


@pytest.fixture
def vector_unconstrained_draws():
    unconstrained_draws = numpy.array([20.2, 10.1, 20.17, 10.08, 20.1, 10.05, 20.13, 10.03]).reshape((4, 1, 2))
    return unconstrained_draws


def test_optimization_fit(scalar_model, scalar_unconstrained_draws):
    fit.OptimizationFit(scalar_model, scalar_unconstrained_draws, tolerance=1e-2)


def test_optimization_fit_error(scalar_model, scalar_unconstrained_draws):
    # Throws an error because the given results are not within tolerance of each other
    with pytest.raises(Exception):
        fit.OptimizationFit(scalar_model, scalar_unconstrained_draws, tolerance=1e-3)


def test_optimization_fit_vector(scalar_model, scalar_unconstrained_draws):
    fit.OptimizationFit(scalar_model, scalar_unconstrained_draws, tolerance=1e-2)


def test_optimization_fit_error_vector(scalar_model, scalar_unconstrained_draws):
    # Throws an error because the given results are not within tolerance of each other
    with pytest.raises(Exception):
        fit.OptimizationFit(scalar_model, scalar_unconstrained_draws, tolerance=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
