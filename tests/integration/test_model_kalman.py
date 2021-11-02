import os
import pathlib
import pandas
import pytest

from rat import ops
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_kalman():
    data_df = pandas.read_csv(os.path.join(test_dir, "kalman.csv"))

    parsed_lines = [
        ops.Normal(
            ops.Data("y"), ops.Param("mu", ops.Index(("i",))), ops.RealConstant(0.1)
        ),
        ops.Normal(
            ops.Param("mu", ops.Index(("i",))),
            ops.Param("mu", ops.Index(("i",), shift_columns=("i",), shift=1)),
            ops.RealConstant(0.3),
        ),
    ]

    model = Model(data_df, parsed_lines)
    fit = model.optimize()
    mu_df = fit.draws("mu")

    mu_ref = [
        -0.4372130,
        -0.2322210,
        -0.1530660,
        -0.6052980,
        -0.8591810,
        -0.6521860,
        -0.4640860,
        -0.0444671,
        -0.1890310,
        0.0243282,
    ]

    assert mu_df["value"].to_list() == pytest.approx(mu_ref, rel=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
