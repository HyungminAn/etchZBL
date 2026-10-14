"""
LAMMPS potential definition blocks.

Each class renders the units/boundary/read_data header plus the pair_style
and pair_coeff lines for a specific potential type.
Add new potential types by subclassing LammpsBlock.
"""
from ase.data import atomic_masses, atomic_numbers

from .base import LammpsBlock
from PlasmaEtchSimulator.params.potentials import get_d2, get_zbl


_LAMMPS_HEADER = """\
units           metal
newton          on
dimension       3
boundary        p p p
atom_style      atomic
box tilt large

read_data       ${strin}
"""


def _mass_lines(elmlist: list) -> str:
    lines = ''
    for idx, elm in enumerate(elmlist):
        atn = atomic_numbers[elm]
        lines += f'mass  {idx + 1} {atomic_masses[atn]:.6f}\n'
    return lines


class BpnnPotentialBlock(LammpsBlock):
    """BPNN (nn/intel) + D2 dispersion + ZBL short-range repulsion. CPU-based."""

    def __init__(self, elmlist: list, potloc: str) -> None:
        self.elmlist = elmlist
        self.potloc = potloc

    def render(self) -> str:
        elm_str = ' '.join(self.elmlist)
        pair_coeff = f'pair_coeff * * nn/intel {self.potloc}  {elm_str}\n'

        for i, e1 in enumerate(self.elmlist):
            for j, e2 in enumerate(self.elmlist[i:], start=i):
                z1, z2 = atomic_numbers[e1], atomic_numbers[e2]
                inner, outer = get_zbl(e1, e2)
                pair_coeff += (
                    f'pair_coeff {i + 1} {j + 1} zbl/pair {z1} {z2} {inner} {outer}\n'
                )

        for i, e1 in enumerate(self.elmlist):
            for j, e2 in enumerate(self.elmlist[i:], start=i):
                c6, r0 = get_d2(e1, e2)
                pair_coeff += f'pair_coeff {i + 1} {j + 1} d2 {c6} {r0}\n'

        return (
            _LAMMPS_HEADER + '\n'
            + _mass_lines(self.elmlist) + '\n'
            + 'pair_style hybrid/overlay nn/intel   d2 15 0.75 20 zbl/pair\n'
            + pair_coeff + '\n'
        )


class E3gnnPotentialBlock(LammpsBlock):
    """E3GNN potential, optionally with D3 dispersion. GPU-accelerated."""

    def __init__(self, elmlist: list, potloc: str, d3: bool = False) -> None:
        self.elmlist = elmlist
        self.potloc = potloc
        self.d3 = d3

    def render(self) -> str:
        elm_str = ' '.join(self.elmlist)
        if self.d3:
            pair_style = 'pair_style hybrid/overlay e3gnn  d3 9000 1600 damp_bj pbe'
            pair_coeff = (
                f'pair_coeff * * e3gnn {self.potloc}  {elm_str}\n'
                f'pair_coeff * * d3  {elm_str}\n'
            )
        else:
            pair_style = 'pair_style e3gnn'
            pair_coeff = f'pair_coeff * *  {self.potloc}  {elm_str}\n'

        return (
            _LAMMPS_HEADER + '\n'
            + _mass_lines(self.elmlist) + '\n'
            + pair_style + '\n'
            + pair_coeff + '\n'
        )


class SevenNetPotentialBlock(LammpsBlock):
    """SevenNet GPU potential via the sevennet pair_style."""

    def __init__(self, elmlist: list, potloc: str) -> None:
        self.elmlist = elmlist
        self.potloc = potloc

    def render(self) -> str:
        elm_str = ' '.join(self.elmlist)
        return (
            _LAMMPS_HEADER + '\n'
            + _mass_lines(self.elmlist) + '\n'
            + 'pair_style sevennet 0\n'
            + f'pair_coeff * * {self.potloc}  {elm_str}\n\n'
        )
