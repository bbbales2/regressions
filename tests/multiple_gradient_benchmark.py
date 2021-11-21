import jax
import jax.numpy
import numpy
import scipy.special
import scipy.stats
import time


N = int(1e6)
M1 = int(1e4)
M2 = int(1e3)

group1 = numpy.random.randint(low=0, high=M1, size=N)
group2 = numpy.random.randint(low=0, high=M2, size=N)

theta1 = numpy.random.rand(M1)
theta2 = numpy.random.rand(M2)

prob = scipy.special.expit(theta1[group1] + theta2[group2])

y = scipy.stats.bernoulli.rvs(prob)

# print(y)
# print(y.dtype)


def target(y, group1, group2, theta1, theta2):
    p = jax.scipy.special.expit(theta1[group1] + theta2[group2])
    return jax.scipy.stats.bernoulli.logpmf(y, p)


def target_vec_params(y, group1, group2, theta1, theta2):
    return jax.vmap(target, (None, None, None, 0, 0))(y, group1, group2, theta1, theta2)


def target_vec_data_vec_params(y, group1, group2, theta1, theta2):
    return jax.vmap(target_vec_params, (0, 0, 0, None, None))(
        y, group1, group2, theta1, theta2
    )


def target_vec_data(y, group1, group2, theta1, theta2):
    return jax.vmap(target, (0, 0, 0, None, None))(y, group1, group2, theta1, theta2)


def target_vec_params_vec_data(y, group1, group2, theta1, theta2):
    return jax.vmap(target_vec_data, (None, None, None, 0, 0))(
        y, group1, group2, theta1, theta2
    )


def single_chain_target(y, group1, group2, theta1, theta2):
    return -jax.numpy.sum(target_vec_data(y, group1, group2, theta1, theta2))


def multiple_chain_target(y, group1, group2, theta1, theta2):
    return -jax.numpy.sum(target_vec_data_vec_params(y, group1, group2, theta1, theta2))


def alternate_multiple_chain_target(y, group1, group2, theta1, theta2):
    return -jax.numpy.sum(target_vec_params_vec_data(y, group1, group2, theta1, theta2))


single_chain_grad = jax.jit(jax.grad(jax.jit(single_chain_target), argnums=(3, 4)))
multiple_chain_grad = jax.jit(jax.grad(jax.jit(multiple_chain_target), argnums=(3, 4)))
alternate_multiple_chain_grad = jax.jit(
    jax.grad(jax.jit(alternate_multiple_chain_target), argnums=(3, 4))
)


def parallel_single_chain_grad(y, group1, group2, theta1, theta2):
    pmapped = jax.pmap(single_chain_grad, in_axes=(0, 0, 0, None, None))
    return tuple(
        jax.numpy.sum(array, axis=0)
        for array in pmapped(y, group1, group2, theta1, theta2)
    )


def parallel_multiple_chain_grad(y, group1, group2, theta1, theta2):
    pmapped = jax.pmap(multiple_chain_grad, in_axes=(0, 0, 0, None, None))
    return tuple(
        jax.numpy.sum(array, axis=0)
        for array in pmapped(y, group1, group2, theta1, theta2)
    )


def parallel_alternate_multiple_chain_grad(y, group1, group2, theta1, theta2):
    pmapped = jax.pmap(alternate_multiple_chain_grad, in_axes=(0, 0, 0, None, None))
    return tuple(
        jax.numpy.sum(array, axis=0)
        for array in pmapped(y, group1, group2, theta1, theta2)
    )


def benchmark(chains=1, devices=1, n_warmup=100, n_time=100):
    print(f"Benchmarking {chains} chains on {devices} devices")
    base_theta1 = jax.device_put(theta1)
    base_theta2 = jax.device_put(theta2)
    stacked_theta1 = jax.device_put(numpy.vstack(chains * [theta1]))
    stacked_theta2 = jax.device_put(numpy.vstack(chains * [theta2]))

    if devices == 1:
        local_y = y
        local_group1 = group1
        local_group2 = group2

        which_single_chain_grad = single_chain_grad
        which_multiple_chain_grad = multiple_chain_grad
        which_alternate_multiple_chain_grad = alternate_multiple_chain_grad
    else:
        local_y = numpy.reshape(y, (devices, -1))
        local_group1 = numpy.reshape(group1, (devices, -1))
        local_group2 = numpy.reshape(group2, (devices, -1))

        which_single_chain_grad = parallel_single_chain_grad
        which_multiple_chain_grad = parallel_multiple_chain_grad
        which_alternate_multiple_chain_grad = parallel_alternate_multiple_chain_grad

    device_y = jax.device_put(local_y)
    device_group1 = jax.device_put(local_group1)
    device_group2 = jax.device_put(local_group2)

    # Do a few gradients to warm up
    for i in range(100):
        gradients_base = which_single_chain_grad(
            device_y, device_group1, device_group2, base_theta1, base_theta2
        )
        gradients_stacked = which_multiple_chain_grad(
            device_y, device_group1, device_group2, stacked_theta1, stacked_theta2
        )
        gradients_alternate_stacked = which_alternate_multiple_chain_grad(
            device_y, device_group1, device_group2, stacked_theta1, stacked_theta2
        )

    # Check all the gradients match
    for g1, g2, g3 in zip(
        gradients_base, gradients_stacked, gradients_alternate_stacked
    ):
        for j in range(chains):
            assert numpy.allclose(g1, g2[j], atol=1e-5)
            assert numpy.allclose(g2[j], g3[j], atol=1e-5)

    # Benchmark the stacked version
    start = time.time()
    for i in range(n_time):
        which_multiple_chain_grad(
            device_y, device_group1, device_group2, stacked_theta1, stacked_theta2
        )
    print(
        f"{(time.time() - start) / n_time}s per gradient stacked (vectorize chains then data)"
    )

    # Benchmark the alternate stacked version
    start = time.time()
    for i in range(n_time):
        which_alternate_multiple_chain_grad(
            device_y, device_group1, device_group2, stacked_theta1, stacked_theta2
        )
    print(
        f"{(time.time() - start) / n_time}s per gradient alternate stacked (vectorize data then chains)"
    )

    # Benchmark the base version
    start = time.time()
    for i in range(n_time):
        for j in range(chains):
            which_single_chain_grad(
                device_y, device_group1, device_group2, base_theta1, base_theta2
            )
    print(
        f"{(time.time() - start) / n_time}s per gradient looped (vectorize only data)"
    )


benchmark(chains=1, devices=1)
benchmark(chains=1, devices=2)
benchmark(chains=1, devices=4)
benchmark(chains=2, devices=1)
benchmark(chains=2, devices=2)
benchmark(chains=2, devices=4)
benchmark(chains=4, devices=1)
benchmark(chains=4, devices=2)
benchmark(chains=4, devices=4)
