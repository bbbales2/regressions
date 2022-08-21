import jax
import numpy
import scipy.optimize

from . import fit


def optimize(model, init=2, chains=4, retries=5, tolerance=1e-2):
    """
    Maximize the log density. `chains` difference optimizations are initialized.

    An error is thrown if the different solutions are not all within tolerance of the
    median solution for each parameter. If only one chain is used, the tolerance is
    ignored.

    If any optimization fails, retry up to `retries` number of times.

    Initialize parameters in unconstrained space uniformly [-2, 2].
    """

    def negative_log_density(x):
        return -model.log_density_no_jac(x.astype(numpy.float32))

    grad = jax.jit(jax.grad(negative_log_density))

    def grad_double(x):
        grad_device_array = grad(x)
        return numpy.array(grad_device_array).astype(numpy.float64)

    unconstrained_draws = numpy.zeros((1, chains, model.size))
    for chain in range(chains):
        for retry in range(retries):
            params = 2 * init * numpy.random.uniform(size=model.size) - init

            solution = scipy.optimize.minimize(negative_log_density, params, jac=grad_double, method="L-BFGS-B",
                                               tol=1e-9)

            if solution.success:
                unconstrained_draws[0, chain] = solution.x
                break
        else:
            raise Exception(f"Optimization failed on chain {chain} with message: {solution.message}")

    constrained_draws, base_dfs = model._prepare_draws_and_dfs(unconstrained_draws)
    return fit.OptimizationFit._from_constrained_variables(constrained_draws, base_dfs, tolerance=tolerance)
