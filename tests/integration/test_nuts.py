import jax
import logging
import numpy
import os
import pathlib
import pandas
import pytest

from rat import nuts
from rat import compiler
from rat.model import Model
from rat.parser import Parser
from pathlib import Path

test_dir = pathlib.Path(__file__).parent


def test_nuts():
    def negative_log_density(q):
        zq = (q - 1.3) / 3.0
        return 0.5 * jax.numpy.dot(zq, zq)

    M_inv = numpy.identity(2)

    ham = nuts.Hamiltonian(negative_log_density, M_inv)

    qs = numpy.zeros((500, M_inv.shape[0]))
    for n in range(1, qs.shape[0]):
        next_draw = nuts.one_sample_nuts(qs[n - 1], 2.1, ham, 0)
        qs[n] = next_draw["q"]
        # print(qs[n])

    print("Finished sampling")
    print("Mean")
    print(numpy.mean(qs, axis=0))
    print("Standard deviation")
    print(numpy.std(qs, axis=0))
    print("Standard error")
    print(numpy.std(qs, axis=0) / numpy.sqrt(qs.shape[0]))


if __name__ == "__main__":
    test_nuts()
#    logging.getLogger().setLevel(logging.DEBUG)
#    pytest.main([__file__, "-s", "-o", "log_cli=true"])
