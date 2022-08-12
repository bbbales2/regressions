import jax
import jax.scipy
from jax.numpy import *


def bernoulli_logit(y, logit_p):
    log_p = -jax.numpy.log1p(jax.numpy.exp(-logit_p))
    log1m_p = -logit_p + log_p
    return jax.numpy.where(y == 0, log1m_p, log_p)


def normal(y, mu, sigma):
    return jax.scipy.stats.norm.logpdf(y, mu, sigma)


def cauchy(y, loc, scale):
    return jax.scipy.stats.cauchy.logpdf(y, loc, scale)


def log_normal(y, mu, sigma):
    logy = jax.numpy.log(y)
    return jax.scipy.stats.norm.logpdf(logy, mu, sigma) - logy


def lax_select_scalar(pred, on_true, on_false):
    # For jax.lax.select, the dimensions pred, on_true, on_false must all match.
    # This may not be true for values return by directly generated code(ex. select([True, False], 0.0, 1.0).
    # This function is a wrapper to support scalar expressions with array preds by expanding scalars to an array,
    # making them compatible with select()
    if isinstance(pred, jax.numpy.ndarray):
        if not isinstance(on_true, jax.numpy.ndarray):
            # if pred is an array but on_true is a scalar, convert on_true to array by filling
            on_true = jax.numpy.full(pred.shape, on_true)
        elif isinstance(on_true, jax.numpy.ndarray) and on_true.size == 1 and len(pred.shape) > 0:
            # do the same for "arrays" pretending to be an array (array with no dims)
            on_true = jax.numpy.tile(on_true, pred.size)

        # do the exact same thing for on_false
        if not isinstance(on_false, jax.numpy.ndarray):
            on_false = jax.numpy.full(pred.shape, on_false)
        elif isinstance(on_false, jax.numpy.ndarray) and on_false.size == 1 and len(pred.shape) > 0:
            on_false = jax.numpy.tile(on_false, pred.size)

    return jax.lax.select(pred, on_true, on_false)
