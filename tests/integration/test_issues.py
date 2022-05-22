import logging
import math
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

issue_data_dir = os.path.join(pathlib.Path(__file__).parent, "issue_data")


def test_ifelse_issue_90():
    data_df = pandas.read_csv(os.path.join(issue_data_dir, "90.csv"))

    with open(os.path.join(issue_data_dir, "90.rat")) as f:
        model_string = f.read()

    Model(data_df, model_string)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
