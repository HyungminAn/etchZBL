import os
from itertools import combinations_with_replacement as H
from itertools import product
from ase import Atom
from ase.io import read, write
from ase.constraints import FixAtoms
import numpy as np


def get_empty_poscar():
    poscar = read('POSCAR')
    while poscar:
        poscar.pop()
    return poscar


def get_xyz_shift(n_bonds):
    xyz_vec_2_bonds = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    xyz_vec_3_bonds = np.array([
        [0.0, 1.0, 0.0],
        [np.sqrt(3)/2, -0.5, 0.0],
        [-np.sqrt(3)/2, -0.5, 0.0],
    ])

    xyz_vec_4_bonds = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, -(np.sqrt(2)-1)],
        [np.sqrt(3)/2, -0.5, -(np.sqrt(2)-1)],
        [-np.sqrt(3)/2, -0.5, -(np.sqrt(2)-1)],
    ])

    if n_bonds == 2:
        xyz_shift = xyz_vec_2_bonds
    elif n_bonds == 3:
        xyz_shift = xyz_vec_3_bonds
    elif n_bonds == 4:
        xyz_shift = xyz_vec_4_bonds

    return xyz_shift


def get_bond_length_matrix():
    bond_length = np.array([
        [2.2824, 1.5291, 1.7255, 1.5421, 1.6354],
        [0.0000, 1.2330, 1.1435, 0.9870, 1.3722],
        [0.0000, 0.0000, 1.3144, 1.1382, 1.2976],
        [0.0000, 0.0000, 0.0000, 0.7503, 0.9381],
        [0.0000, 0.0000, 0.0000, 0.0000, 1.4243],
    ])
    n_rows, n_cols = bond_length.shape
    for row in range(n_rows):
        for col in range(row):
            bond_length[row, col] = \
                bond_length[col, row]
    return bond_length


def gen_mol_with_given_bonds(
        center_atom, bond_atoms, poscar_baseline,
        center_xyz, xyz_shift, n_bonds,
        ):
    bl_mat = get_bond_length_matrix()
    elem_idx_dict = {
        'Si': 0,
        'O': 1,
        'C': 2,
        'H': 3,
        'F': 4,
    }
    for idx, atom_list in enumerate(H(bond_atoms, n_bonds)):
        poscar = poscar_baseline.copy()
        poscar.append(
            Atom(center_atom, center_xyz)
            )

        for atom_type, shift in zip(atom_list, xyz_shift):
            idx_elem1 = elem_idx_dict[center_atom]
            idx_elem2 = elem_idx_dict[atom_type]
            bl = bl_mat[idx_elem1, idx_elem2]
            poscar.append(
                Atom(atom_type, center_xyz + bl*shift)
                )

        c = FixAtoms(indices=[0])
        poscar.set_constraint(c)

        name = poscar.get_chemical_formula()
        os.makedirs(name, exist_ok=True)
        write(f'{name}/POSCAR', poscar, format='vasp', sort=True)
        print(f'{name} written')


def main():
    poscar_baseline = get_empty_poscar()

    center_atom_list = ['O', 'Si', 'C']
    bond_atoms = ['O', 'H', 'F']

    cell_size = poscar_baseline.get_cell()[0, 0]
    center_xyz = np.full(3, cell_size / 2)

    n_bonds_list = [2, 3, 4]
    for center_atom, n_bonds in product(center_atom_list, n_bonds_list):
        if center_atom == 'O' and n_bonds > 2:
            continue

        xyz_shift = get_xyz_shift(n_bonds)

        gen_mol_with_given_bonds(
            center_atom, bond_atoms, poscar_baseline,
            center_xyz, xyz_shift, n_bonds)


main()
