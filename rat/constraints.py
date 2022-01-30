import jax.numpy
import jax.scipy


def lower(y, lower):
    return jax.numpy.exp(y) + lower, jax.numpy.sum(y)


def upper(y, upper):
    return upper - jax.numpy.exp(y), jax.numpy.sum(y)


def finite(y, upper, lower):
    inv_logit_y = jax.scipy.special.expit(y)
    return (
        lower + (upper - lower) * inv_logit_y,
        jax.numpy.sum((upper - lower) * inv_logit_y * (1 - inv_logit_y)),
    )


def offset_multiply(y, offset, multiplier):
    log_multiplier = jax.numpy.log(multiplier)
    if multiplier.size == 1:
        jacobian_adjustment = y.size * log_multiplier
    else:
        jacobian_adjustment = jax.numpy.sum(log_multiplier)

    return y * multiplier + offset, jacobian_adjustment
