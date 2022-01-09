import jax

def bernoulli_logit(y, logit_p):
    log_p = -jax.numpy.log1p(jax.numpy.exp(-logit_p))
    log1m_p = -logit_p + log_p
    return jax.numpy.where(y == 0, log1m_p, log_p)


def log_normal(y, mu, sigma):
    logy = jax.numpy.log(y)
    return jax.scipy.stats.norm.logpdf(logy, mu, sigma) - logy
