from concurrent.futures import ThreadPoolExecutor
import jax
import logging
import numpy
import os
import pathlib
import pandas
import pytest
import plotnine
import time

from rat import nuts
from rat import compiler
from rat.model import Model
from rat.parser import Parser
from pathlib import Path

test_dir = pathlib.Path(__file__).parent


def test_nuts():
    def negative_log_density(q):
        return -jax.numpy.sum(jax.scipy.stats.norm.logpdf(q, loc=17.1, scale=jax.numpy.array([0.3, 1.4])))

    potential = nuts.Potential(negative_log_density, 1, 2)
    rng = numpy.random.default_rng()

    initial_draw = numpy.zeros(2)

    draw, stepsize, diagonal_inverse_metric = nuts.warmup(potential, rng, initial_draw)

    # qs = numpy.zeros((100, diag_M_inv.shape[0]))
    # for n in range(1, qs.shape[0]):
    #    next_draw = nuts.one_draw(potential, rng, qs[n - 1], 0.1, diag_M_inv, debug = True)
    #    qs[n] = next_draw["q"]
    #    print(qs[n], next_draw["accept_stat"])

    draws = nuts.sample(potential, rng, draw, stepsize, diagonal_inverse_metric, 10000)

    means = numpy.mean(draws, axis=0)
    stds = numpy.std(draws, axis=0)

    assert means == pytest.approx([17.1, 17.1], rel=0.05)
    assert stds == pytest.approx([0.3, 1.4], rel=0.10)


def test_multithreaded_nuts():
    def negative_log_density(q):
        return -jax.numpy.sum(jax.scipy.stats.norm.logpdf(q, loc=17.1, scale=jax.numpy.array([0.3, 1.4])))

    chains = 4
    size = 2

    def generate_draws():
        rng = numpy.random.default_rng()
        initial_draw = numpy.zeros(size)

        draw, stepsize, diagonal_inverse_metric = nuts.warmup(potential, rng, initial_draw)
        return nuts.sample(potential, rng, draw, stepsize, diagonal_inverse_metric, 1000)

    start = time.time()
    with ThreadPoolExecutor(max_workers=chains) as e:
        potential = nuts.Potential(negative_log_density, chains, size)

        results = []
        for chain in range(chains):
            results.append(e.submit(generate_draws))

        draws = numpy.array([result.result() for result in results])
    print("Total time: ", time.time() - start)
    print("Total gradient time: ", sum(N * gradient_time for N, gradient_time in potential.metrics.values()))

    for count, (N, gradient_time) in potential.metrics.items():
        print("Time for gradients: ", count, N, gradient_time, gradient_time * N)

    means = numpy.mean(draws, axis=(0, 1))
    stds = numpy.std(draws, axis=(0, 1))

    plot = plotnine.qplot(range(draws.shape[1]), draws[0, :, 0])
    plot.draw(show = True)

def test_one_draw():
    def negative_log_density(q):
        return 0.5 * jax.numpy.dot(q, q)

    chains = 1
    size = 2

    rng = numpy.random.default_rng(seed = 5)
    initial_draw = numpy.ones(size)

    potential = nuts.Potential(negative_log_density, chains, size)

    nuts.one_draw(potential, rng, initial_draw, 0.005, numpy.array([1.3, 1.7]))


if __name__ == "__main__":
    #test_nuts()
    test_multithreaded_nuts()
    #test_one_draw()
    # logging.getLogger().setLevel(logging.DEBUG)
    # pytest.main([__file__, "-s", "-o", "log_cli=true"])
