import logging
import os
import pathlib
import pandas
import pytest

from rat import ops
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_noncenter():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = """
    y ~ normal(theta[school], sigma[school]);
    theta[school] = mu + z[school] * tau;
    z[school] ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ lognormal(0, 1);
    """

    # TODO: Add a unit test that the thing above parses to the thing below
    parsed_lines = [
        ops.Normal(ops.Data("y"), ops.Param("theta", ops.Index(("school",))), ops.Data("sigma")),
        ops.Assignment(
            ops.Param("theta", ops.Index(("school",))),
            ops.Sum(ops.Param("mu"), ops.Mul(ops.Param("z", ops.Index(("school",))), ops.Param("tau"))),
        ),
        ops.Normal(ops.Param("mu"), ops.RealConstant(0.0), ops.RealConstant(5.0)),
        ops.Normal(ops.Param("z", ops.Index(("school",))), ops.RealConstant(0.0), ops.RealConstant(1.0)),
        ops.LogNormal(ops.Param("tau", lower=ops.RealConstant(0.0)), ops.RealConstant(0.0), ops.RealConstant(1.0)),
    ]

    model = Model(data_df, parsed_lines)  # model_string=model_string
    fit = model.optimize(init=0.1)
    mu_df = fit.draws("mu")
    z_df = fit.draws("z")
    tau_df = fit.draws("tau")

    ref_z = [
        0.03839950,
        0.01248300,
        -0.01099810,
        0.00726588,
        -0.02560940,
        -0.01104690,
        0.04940930,
        0.00841920,
    ]
    assert mu_df["value"][0] == pytest.approx(4.61934000, rel=1e-2)
    assert tau_df["value"][0] == pytest.approx(0.36975800, rel=1e-2)
    assert z_df["value"].to_list() == pytest.approx(ref_z, rel=1e-2, abs=1e-3)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])