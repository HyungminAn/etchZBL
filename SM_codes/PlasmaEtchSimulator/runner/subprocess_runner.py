"""
Direct subprocess runner: blocks until LAMMPS exits and checks the return code.
"""
import subprocess

from .base import RunnerBase


class SubprocessRunner(RunnerBase):
    """Run LAMMPS as a blocking subprocess and check its exit code."""

    def run(self, cmd: str, log_file: 'str | None' = None) -> bool:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0
