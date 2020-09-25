# Copyright (c) 2020 Graphcore Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been added by Graphcore Ltd.

"""Tests for the IPU interface to Minigo"""
import os
import pytest
from pathlib import Path

import unittest
from typing import List
import subprocess


class SubProcessChecker(unittest.TestCase):
    """
    Utility Module for building tests that reliably check if a
    sub-process ran successfully.

    Commonly with an integration/system test you want to check
    a command can be run successfully and gives some expected
    output.

    How to use:
    1. Make a test case in the normal way but inherit from
    test_util.SubProcessChecker instead of unittest.TestCase.
    2. Define a test method in your derived class in the normal way.
    3. Have the test method call self.run_command(...) and the output
    will be checked automatically.

    .. note:: Copied from the Graphcore examples repository.
    """

    def _check_output(self, cmd, output: str, must_contain: List[str]):
        """
        Internal utility used by run_command(...) to check output
        (Should not need to call this directly from test cases).
        """
        if not must_contain:
            return
        # If a string is passed in convert it to a list
        if isinstance(must_contain, str):
            must_contain = [must_contain]
        # Build a list of regexes then apply them all.
        # Each must have at least one result:
        regexes = [re.compile(s) for s in must_contain]
        for i, r in enumerate(regexes):
            match = r.search(output)
            if not match:
                self.fail(f"Output of command: '{cmd}' contained no match for: '{must_contain[i]}'\nOutput was:\n{output}")

    def run_command(self, cmd, working_path, expected_strings, env=None, timeout=None):
        """
        Run a command using subprocess, check it ran successfully, and
        check its output.

        Args:
            cmd:
                Command string. It will be split into args internally.
            working_path:
                The working directory in which to run the command.
            expected_strings:
                List of strings that must appear in the output at least once.
            env:
                Optionally pass in the Environment variables to use
            timeout:
                Optionally pass in the timeout for running the command
            Returns:
                Output of the command (combined stderr and stdout).
        """
        if env is None:
            completed = subprocess.run(args=cmd.split(),
                                       cwd=working_path,
                                       shell=False,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       timeout=timeout)
        else:
            completed = subprocess.run(args=cmd.split(), cwd=working_path,
                                       shell=False, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       env=env,
                                       timeout=timeout)
        combined_output = str(completed.stdout, 'utf-8')
        try:
            completed.check_returncode()
            return_code_ok = True
        except subprocess.CalledProcessError:
            return_code_ok = False

        if not return_code_ok:
            self.fail(f"The following command failed: {cmd}\nWorking path: {working_path}\nOutput of failed command:\n{combined_output}")

        self._check_output(cmd, combined_output, expected_strings)
        return combined_output


working_path = Path(__file__).parent.joinpath("minigo")
current_path = Path(__file__).parent


class SetUpTest(SubProcessChecker):
    """Separately test the setup routines."""
    def test_ipu_reservation_file_creation(self):
        """IPU reservation graph files creation"""
        # test ipu config file creation
        self.run_command(
           "python ipu_reservation_graph.py", current_path, [])

    def test_data_load_and_file_freeze(self):
        """Data load and file freeze"""
        # test ipu data loading
        self.run_command(
          "python ml_perf/get_data.py", working_path, [])


class BenchmarkTest(SubProcessChecker):
    """Test all core functions of the minigo benchmark with reduced workload"""
    def setUp(self):
        """Compile the c++ code, create configs, and download data."""

        self.run_command(
          "python ipu_reservation_graph.py", current_path, [])

        self.run_command(
          "python ml_perf/get_data.py", working_path, [])

    def test_single_round(self):
        """Test the benchmark for the initial and an additional loop"""
        # test ipu config file creation
        self.run_command(
            "python ipu_reservation_graph.py", current_path, [])
        # test ipu data loading
        self.run_command(
            "python ml_perf/get_data.py", working_path, [])

        # test main benchmark with only one iteration
        pwd = os.getcwd()
        BASEDIR = os.sep.join([pwd, "minigo", "results", "single_round_test"])
        self.run_command(
            "python ml_perf/reference_implementation.py --base_dir={} --flagfile=ml_perf/flags/9/test_rl_loop.flags".format(BASEDIR),
            working_path, [])

    def test_evaluation_round(self):
        """Run single benchmark round and evaluate afterwards."""
        # test ipu config file creation
        self.run_command(
            "python ipu_reservation_graph.py", current_path, [])
        # test ipu data loading
        self.run_command(
            "python ml_perf/get_data.py", working_path, [])

        pwd = os.getcwd()
        BASEDIR = os.sep.join([pwd, "minigo", "results", "evaluation_round_test"])
        self.run_command(
            "python ml_perf/reference_implementation.py --base_dir={} --flagfile=ml_perf/flags/9/test_rl_loop.flags".format(BASEDIR),
            working_path, [])

        self.run_command(
            "python ml_perf/eval_models.py --base_dir={} --flags_dir=ml_perf/flags/9/".format(BASEDIR),
            working_path, [])