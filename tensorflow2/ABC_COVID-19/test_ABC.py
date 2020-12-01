# Copyright 2020 Graphcore Ltd.
"""Tests for the COVID-19 ABC algorithm"""
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
        import re
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


current_path = Path(__file__).parent


class AbcTest(SubProcessChecker):
    """Test simple command line executions"""

    def test_default(self):
        """Test the defaults"""
        self.run_command(
            "python ABC_IPU.py", current_path, ["Time per run"])

    def test_italy(self):
        self.run_command(
            "python ABC_IPU.py -cn Italy -t 2e5 -b 100 -s 1",
            current_path, ["Time per run"])

    def test_nz(self):
        self.run_command(
            'python ABC_IPU.py -cn New_Zealand -t 1e5 -b 100 -s 1',
            current_path, ["Time per run"])

    def test_usa(self):
        self.run_command(
            "python ABC_IPU.py -cn US -t 1e6 -b 100 -s 1",
            current_path, ["Time per run"])

    def test_replication(self):
        self.run_command(
            "python ABC_IPU.py -r 2 -b 100 -s 1",
            current_path, ["Time per run"])

    def test_readme(self):
        self.run_command(
            "python ABC_IPU.py --enqueue-chunk-size 10000 --tolerance 5e5 "
            "--n-samples-target 100 --n-samples-per-batch 100000 --country US"
            " --samples-filepath US_5e5_100.txt",
            current_path, ["Time per run"])
