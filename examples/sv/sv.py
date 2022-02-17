import rat.constraints
import rat.math
import jax

unconstrained_parameter_size = 7


def constrain_parameters(unconstrained_parameter_vector, pad=True):
    unconstrained_parameters = {}
    parameters = {}
    total = 0.0

    # sigma
    unconstrained_parameters["sigma"] = unconstrained_parameter_vector[..., 0]
    parameters["sigma"], constraints_jacobian_adjustment = rat.constraints.lower(unconstrained_parameters["sigma"], 0.0)
    total += jax.numpy.sum(constraints_jacobian_adjustment)

    # v_raw
    unconstrained_parameters["v_raw"] = unconstrained_parameter_vector[..., 1:3]
    parameters["v_raw"] = unconstrained_parameters["v_raw"]

    # rho
    unconstrained_parameters["rho"] = unconstrained_parameter_vector[..., 3]
    parameters["rho"] = unconstrained_parameters["rho"]

    # mu
    unconstrained_parameters["mu"] = unconstrained_parameter_vector[..., 4]
    parameters["mu"] = unconstrained_parameters["mu"]

    # mu_ret
    unconstrained_parameters["mu_ret"] = unconstrained_parameter_vector[..., 5]
    parameters["mu_ret"] = unconstrained_parameters["mu_ret"]

    # phi_raw
    unconstrained_parameters["phi_raw"] = unconstrained_parameter_vector[..., 6]
    parameters["phi_raw"], constraints_jacobian_adjustment = rat.constraints.finite(unconstrained_parameters["phi_raw"], 0.0, 1.0)
    total += jax.numpy.sum(constraints_jacobian_adjustment)

    return total, parameters


def transform_parameters(data, subscripts, parameters):
    # v
    parameters["v"] = parameters["sigma"] * parameters["v_raw"][subscripts["t__2"]]

    # phi
    parameters["phi"] = 2.0 * parameters["phi"] - 1.0

    # h
    # I'm not sure how to handle the separate definitions of h[0] and h[t]. My guess is we build them separately and concatenate,
    # so that is what I have done here
    #
    # I am treating h[0] like it is a scalar (makes the carry cleaner) -- this implies subscripts['t__10'] is an integer (not a list of integers)
    parameters["h[0]"] = parameters["v"][subscripts["t__10"]] / jax.numpy.sqrt(1 - parameters["phi"] * parameters["phi"])

    # A recursive expression that had more dependencies would simply need a longer carry
    def recursion_h(carry, v):
        # The carry here is a length 1 tuple containing the last value of h
        h = parameters["phi"] * carry[0] + v
        # The left hand side of the return is the carry (we'll need to save the last value of h for next iteration) and
        # the right hand side is what gets written in the output array
        return (h,), h

    _, parameters["h[t]"] = jax.lax.scan(recursion_h, (parameters["h[0]"],), parameters["v"][subscripts["t__5"]])

    # Bring everything back together -- it may be faster to allocate h as zeros and then add in these other things, but
    # anyway this is something that should work
    parameters["h"] = jax.numpy.concat([parameters["h[t]"], jax.numpy.array([parameters["h[0]"]])])

    # s
    parameters["s"] = jax.numpy.exp(parameters["mu"] + parameters["h"][subscripts["t__7"]] / 2.0)

    return parameters


def evaluate_densities(data, subscripts, parameters):
    target = 0.0
    target += jax.numpy.sum(jax.scipy.stats.expon.logpdf(parameters["sigma"], loc=0, scale=10.0))
    target += jax.numpy.sum(jax.scipy.stats.norm.logpdf(parameters["v_raw"][subscripts["t__0"]], 0.0, 1.0))
    target += jax.numpy.sum(jax.scipy.stats.norm.logpdf(parameters["rho"], -1.0, 1.0))
    target += jax.numpy.sum(jax.scipy.stats.norm.logpdf(parameters["mu"], 0.0, 1.0))
    target += jax.numpy.sum(jax.scipy.stats.norm.logpdf(parameters["mu_ret"], 0.0, 0.2))
    target += jax.numpy.sum(
        jax.scipy.stats.norm.logpdf(
            data["y"],
            parameters["mu_ret"] + parameters["rho"] * parameters["v"][subscripts["t__8"]] * parameters["s"][subscripts["t__9"]],
            parameters["s"][subscripts["t__10"]],
        )
    )
    target += jax.numpy.sum(jax.scipy.stats.norm.logpdf(parameters["phi_raw"], 0.0, 1.0))

    return target
