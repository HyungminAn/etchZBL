"""
Abstract base class for LAMMPS execution backends.
"""
from abc import ABC, abstractmethod


class RunnerBase(ABC):
    """Execute a LAMMPS command and report success or failure."""

    @abstractmethod
    def run(self, cmd: str, log_file: 'str | None' = None) -> bool:
        """Execute the given shell command. Returns True on success."""
