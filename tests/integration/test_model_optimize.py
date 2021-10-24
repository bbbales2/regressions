import os
import pathlib
import pandas
import pytest

from rat import ops
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_normal_mu():
    data_df = pandas.read_csv(os.path.join(test_dir, "normal.csv"))

    parsed_lines = [
        ops.Normal(ops.Data("y"), ops.Param("mu"), ops.RealConstant(1.5)),
        ops.Normal(
            ops.Param("mu"), ops.RealConstant(-0.5), ops.RealConstant(0.3)
        ),
    ]

    model = Model(data_df, parsed_lines)
    fit = model.optimize()
    mu_df = fit.draws("mu")

    assert mu_df["mu"][0] == pytest.approx(-1.11249, rel = 1e-4)
