"""
LAMMPS molecule file writer for ion deposition.
"""
import re

from PlasmaEtchSimulator.params.molecules import ION_GEOMETRIES


def write_ion_file(path: str, ion: str, elmlist: list) -> None:
    """Write a LAMMPS molecule file for the given ion species."""
    if ion not in ION_GEOMETRIES:
        raise ValueError(
            f"Ion '{ion}' is not defined in params/molecules.py::ION_GEOMETRIES. "
            f"Available ions: {list(ION_GEOMETRIES)}"
        )

    entries = ION_GEOMETRIES[ion]
    n_atoms = len(entries)

    coords_lines = ''
    types_lines = ''
    for count, (elm, pos) in enumerate(entries, start=1):
        coords_lines += f'{count} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n'
        atom_type = elmlist.index(elm) + 1
        types_lines += f'{count} {atom_type}\n'

    content = (
        f'# Molecule {ion}\n'
        f'{n_atoms} atoms\n\n'
        f'Coords\n\n'
        f'{coords_lines}\n'
        f'Types\n\n'
        f'{types_lines}'
    )

    with open(path, 'w') as f:
        f.write(content)
    print(f'Ion file written: {path}')
