import jax.numpy
import jax.scipy

def lower(y, lower):
    return jax.numpy.exp(y) + lower, jax.numpy.sum(y)

def upper(y, upper):
    return upper - jax.numpy.exp(y), jax.numpy.sum(y)

def finite(y, upper, lower):
    inv_logit_y = jax.scipy.expit(y)
    return (
        lower + (upper - lower) * inv_logit_y,
        jax.numpy.sum((upper - lower) * inv_logit_y * (1 - inv_logit_y))
    )
