"""
Polling runner: submits the job and polls the LAMMPS log file until completion.

Use this when LAMMPS must be submitted via a scheduler (e.g. PBS, Slurm)
or when you cannot block the Python process directly. The runner monitors
log_file for the 'Total wall time' line that LAMMPS writes at the end of a run.
"""
import os
import subprocess
import time

from .base import RunnerBase

_COMPLETION_MARKER = 'Total wall time'


class PollingRunner(RunnerBase):
    """
    Submit a job with subprocess.Popen, then poll the LAMMPS log file every
    poll_interval seconds until the completion marker appears or a timeout.
    """

    def __init__(self, poll_interval: int = 10, timeout: 'int | None' = None) -> None:
        self.poll_interval = poll_interval
        self.timeout = timeout

    def run(self, cmd: str, log_file: 'str | None' = None) -> bool:
        if log_file is None:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0

        if os.path.exists(log_file):
            os.remove(log_file)

        subprocess.Popen(cmd, shell=True)

        elapsed = 0
        while True:
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval
            if self._is_complete(log_file):
                return True
            if self.timeout is not None and elapsed >= self.timeout:
                return False

    @staticmethod
    def _is_complete(log_file: str) -> bool:
        """Return True when log_file exists and contains the completion marker."""
        if not os.path.exists(log_file):
            return False
        try:
            with open(log_file) as f:
                return _COMPLETION_MARKER in f.read()
        except OSError:
            return False
