import logging
import numpy
import os
import pathlib
import pandas
import pytest
import tempfile

from rat.model import Model
from rat import optimize
from pathlib import Path

test_dir = pathlib.Path(__file__).parent


def test_model_compile_path():
    data_df = pandas.read_csv(os.path.join(test_dir, "../integration/normal.csv"))

    model_string = """
    y[n]' ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    model = Model(model_string=model_string, data=data_df)
    optimize(model)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
