import os
import pathlib
import pandas
import pytest

from rat import ops
from rat.model import Model

test_dir = pathlib.Path(__file__).parent


def test_optimize_missing_prior_or_assign():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    parsed_lines = [
        ops.Normal(ops.Data("y"), ops.Param("theta", ops.Index(("school",))), ops.Data("sigma")),
    ]

    with pytest.raises(Exception):
        model = Model(data_df, parsed_lines)


def test_optimize_unknown_subscript():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    parsed_lines = [
        ops.Normal(ops.Data("y"), ops.Param("theta", ops.Index(("rabbit",))), ops.Data("sigma")),
    ]

    with pytest.raises(Exception):
        model = Model(data_df, parsed_lines)


def test_optimize_missing_subscript():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    parsed_lines = [
        ops.Normal(ops.Data("y"), ops.Param("theta", ops.Index(("school",))), ops.Data("sigma")),
        ops.Normal(ops.Param("theta"), ops.RealConstant(0.0), ops.RealConstant(5.0)),
    ]

    with pytest.raises(Exception):
        model = Model(data_df, parsed_lines)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
