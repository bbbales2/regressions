import logging
import numpy
import os
import pathlib
import pandas
import pytest
import tempfile

from rat import compiler
from rat.model import Model
from rat.parser import Parser
from pathlib import Path

test_dir = pathlib.Path(__file__).parent


def test_model_compile_path():
    data_df = pandas.read_csv(os.path.join(test_dir, "../integration/normal.csv"))

    model_string = """
    y ~ normal(mu, 1.5);
    mu ~ normal(-0.5, 0.3);
    """

    with tempfile.TemporaryDirectory(prefix="rat.") as working_dir:
        model_path = os.path.join(working_dir, "model_path.py")
        model = Model(data_df, model_string=model_string)

        assert not os.path.exists(model_path)

        model = Model(data_df, model_string=model_string, compile_path = model_path)

        assert os.path.exists(model_path)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
