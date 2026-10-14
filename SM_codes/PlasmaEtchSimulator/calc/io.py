"""
Atomic structure file I/O utilities.
"""
import os
import copy

import ase
import ase.io


def save_atom(atoms: ase.Atoms, fname: str, elmlist: list) -> None:
    """Write an ASE Atoms object to LAMMPS data format."""
    opts = {
        'format': 'lammps-data',
        'specorder': elmlist,
        'velocities': True,
    }
    try:
        ase.io.write(fname, atoms, **opts)
    except Exception:
        opts.pop('velocities')
        ase.io.write(fname, atoms, **opts)
    assert os.path.exists(fname), f'Failed to write {fname}'


def load_atom(fname: str, convert_dict: dict) -> ase.Atoms:
    """Read a LAMMPS data file and remap type indices to element symbols."""
    atoms = ase.io.read(fname, format='lammps-data', sort_by_id=False)
    chem_sym = set(atoms.get_chemical_symbols())
    if chem_sym.issubset(convert_dict.keys()):
        for atom in atoms:
            atom.symbol = convert_dict[atom.symbol]
    positions = atoms.positions.copy()
    for i in range(3):
        positions[:, i] = positions[:, i] % atoms.cell[i, i]
    atoms.positions = positions
    return atoms


def expand_box(atoms: ase.Atoms, box_height: float) -> ase.Atoms:
    """Return a copy of atoms with the Z cell dimension set to box_height."""
    out = copy.deepcopy(atoms)
    positions = out.positions.copy()
    for i in range(3):
        positions[:, i] = positions[:, i] % out.cell[i, i]
    out.cell[2, 2] = box_height
    out.positions = positions
    return out


def make_convert_dict(elmlist: list) -> dict:
    """Map ASE 1-indexed chemical symbols to user element symbols."""
    from ase.data import chemical_symbols
    return {chemical_symbols[idx + 1]: sym for idx, sym in enumerate(elmlist)}
