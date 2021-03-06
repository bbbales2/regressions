import logging
import numpy
import os
import pathlib
import pandas
import pytest
import tempfile

from rat import ast
from rat import compiler
from rat.model import Model
from rat.parser import Parser
from rat.fit import load
from pathlib import Path
import shlex
import subprocess
import sys

test_dir = pathlib.Path(__file__).parent
rat_path = os.path.join(test_dir, "../../bin/rat")


def test_cli_optimize():
    model_filename = os.path.join(test_dir, "eight_schools.rat")
    data_filename = os.path.join(test_dir, "eight_schools.csv")

    cmd_defaults = f"{sys.executable} {rat_path} {model_filename} {{output_folder}} {data_filename} --compile_path={{output_folder}}/model.py --method=optimize --chains=1"
    cmd_specific = cmd_defaults + f" --init=0.1 --retries=5 --tolerance=1e-1 --overwrite"

    for cmd in [cmd_defaults, cmd_specific]:
        with tempfile.TemporaryDirectory() as output_folder:
            output_folder = os.path.join(output_folder, "output")

            result = subprocess.run(cmd.format(output_folder=output_folder), shell=True, capture_output=True)

            try:
                result.check_returncode()
            except subprocess.CalledProcessError as e:
                print("Stdout: ")
                print(result.stdout.decode("utf-8"))
                print("Stderr: ")
                print(result.stderr.decode("utf-8"))
                raise e

            assert os.path.exists(os.path.join(output_folder, "model.py"))


def test_cli_sample():
    model_filename = os.path.join(test_dir, "eight_schools.rat")
    data_filename = os.path.join(test_dir, "eight_schools.csv")

    cmd_defaults = f"{sys.executable} {rat_path} {model_filename} {{output_folder}} data={data_filename} --method=sample --chains=4"
    cmd_specific = cmd_defaults + f" --init=0.1 --num_warmup=400 --num_draws=400 --thin=2 --target_acceptance_rate=0.85 --overwrite"

    for cmd in [cmd_defaults, cmd_specific]:
        with tempfile.TemporaryDirectory() as output_folder:
            output_folder = os.path.join(output_folder, "output")

            subprocess.run(cmd.format(output_folder=output_folder), shell=True, check=True, capture_output=True)

            load(output_folder)


def test_cli_multiple_files():
    model_filename = os.path.join(test_dir, "eight_schools.rat")
    y_data_filename = os.path.join(test_dir, "eight_schools_y.csv")
    sigma_data_filename = os.path.join(test_dir, "eight_schools_sigma.csv")

    cmd = f"{sys.executable} {rat_path} {model_filename} {{output_folder}} y_data={y_data_filename} sigma_data={sigma_data_filename} --compile_path={{output_folder}}/model.py --method=optimize --chains=1"
    with tempfile.TemporaryDirectory() as output_folder:
        output_folder = os.path.join(output_folder, "output")

        result = subprocess.run(cmd.format(output_folder=output_folder), shell=True, capture_output=True)

        try:
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            print("Stdout: ")
            print(result.stdout.decode("utf-8"))
            print("Stderr: ")
            print(result.stderr.decode("utf-8"))
            raise e

        assert os.path.exists(os.path.join(output_folder, "model.py"))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
