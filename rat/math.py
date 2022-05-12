import jax


def bernoulli_logit(y, logit_p):
    log_p = -jax.numpy.log1p(jax.numpy.exp(-logit_p))
    log1m_p = -logit_p + log_p
    return jax.numpy.where(y == 0, log1m_p, log_p)


def log_normal(y, mu, sigma):
    logy = jax.numpy.log(y)
    return jax.scipy.stats.norm.logpdf(logy, mu, sigma) - logy


def lax_select_scalar(pred, on_true, on_false):
    # For jax.lax.select, the dimensions pred, on_true, on_false must all match.
    # This may not be true for values return by directly generated code.
    # This function is a wrapper to support scalar expressions with array preds
    if isinstance(pred, jax.numpy.ndarray):
        if not isinstance(on_true, jax.numpy.ndarray):
            on_true = jax.numpy.full(pred.shape, on_true)
        if not isinstance(on_false, jax.numpy.ndarray):
            on_false = jax.numpy.full(pred.shape, on_false)
    return jax.lax.select(pred, on_true, on_false)
