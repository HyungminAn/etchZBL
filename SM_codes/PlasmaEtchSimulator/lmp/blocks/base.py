"""
Abstract base class for composable LAMMPS script blocks.
"""
from abc import ABC, abstractmethod


class LammpsBlock(ABC):
    """A single composable section of a LAMMPS input script."""

    @abstractmethod
    def render(self) -> str:
        """Return the LAMMPS script text for this block."""
