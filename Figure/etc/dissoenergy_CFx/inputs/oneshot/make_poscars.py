import os
from ase.io import read, write
import numpy as np


def main(src):
    poscar = read(src, format="lammps-data", atom_style="atomic")
    cell = poscar.get_cell()
    cell[0][0] = cell[1][1] = cell[2][2] = 20.0
    poscar.set_cell(cell)

    idx_C = [atom.index for atom in poscar if atom.symbol == "C"][0]
    idx_F = [atom.index for atom in poscar if atom.symbol == "F"][0]

    vec_CF = poscar[idx_F].position - poscar[idx_C].position
    bond_length = np.linalg.norm(vec_CF)
    vec_CF /= bond_length

    shifts = np.arange(0, 6.0-bond_length, 0.1)
    pos_F_original = poscar[idx_F].position.copy()
    for idx, shift in enumerate(shifts):
        poscar[idx_F].position = pos_F_original + shift * vec_CF
        dst = f"shift_{shift:.1f}"
        os.makedirs(dst, exist_ok=True)
        write(f"{dst}/input.data", poscar, format="lammps-data",
              specorder=["Si", "O", "C", "H", "F"])
        write(f"{dst}/POSCAR", poscar, format="vasp")
        print(f"Generated {dst}/input.data, {dst}/POSCAR")


if __name__ == "__main__":
    src = "../inputs/relax_mol/CF/relaxed.coo"
    main(src)
