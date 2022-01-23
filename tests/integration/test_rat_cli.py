import logging
import numpy
import os
import pathlib
import pandas
import pytest
import tempfile

from rat import ops
from rat import compiler
from rat.model import Model
from rat.parser import Parser
from rat.fit import load
from pathlib import Path
import shlex
import subprocess

test_dir = pathlib.Path(__file__).parent
rat_path = os.path.join(test_dir, "../../bin/rat")


def test_cli_optimize():
    model_filename = os.path.join(test_dir, "eight_schools.rat")
    data_filename = os.path.join(test_dir, "eight_schools.csv")

    cmd_defaults = f"{rat_path} {model_filename} {data_filename} {{output_folder}} --method=optimize --chains=1"
    cmd_specific = cmd_defaults + f" --init=0.1 --retries=5 --tolerance=1e-1 --overwrite"

    for cmd in [cmd_defaults, cmd_specific]:
        with tempfile.TemporaryDirectory() as output_folder:
            output_folder = os.path.join(output_folder, "output")

            subprocess.run(cmd.format(output_folder=output_folder), shell=True, check=True)

            load(output_folder)


def test_cli_sample():
    model_filename = os.path.join(test_dir, "eight_schools.rat")
    data_filename = os.path.join(test_dir, "eight_schools.csv")

    cmd_defaults = f"{rat_path} {model_filename} {data_filename} {{output_folder}} --method=sample --chains=4"
    cmd_specific = cmd_defaults + f" --init=0.1 --num_warmup=400 --num_draws=400 --target_acceptance_rate=0.85 --overwrite"

    for cmd in [cmd_defaults, cmd_specific]:
        with tempfile.TemporaryDirectory() as output_folder:
            output_folder = os.path.join(output_folder, "output")

            subprocess.run(cmd.format(output_folder=output_folder), shell=True, check=True)

            load(output_folder)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
