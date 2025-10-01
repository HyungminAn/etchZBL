import os
import numpy as np
from ase.io import read


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


def main():
    path_list = [f'{i}/CONTCAR' for i in os.listdir() if os.path.isdir(i)]

    bl_mat = get_bond_length_matrix()
    elem_idx_dict = {
        'Si': 0,
        'O': 1,
        'C': 2,
        'H': 3,
        'F': 4,
    }

    for path_contcar in path_list:
        contcar = read(path_contcar)
        idx_fixed = contcar._constraints[0].index[0]
        idx_unfixed = [i for i in range(len(contcar)) if i != idx_fixed]
        dist = contcar.get_distances(idx_fixed, idx_unfixed, mic=True)
        atom_types = contcar.get_chemical_symbols()

        for idx, bl_after in zip(idx_unfixed, dist):
            elem1, elem2 = atom_types[idx_fixed], atom_types[idx]
            idx1, idx2 = elem_idx_dict[elem1], elem_idx_dict[elem2]
            bl_before = bl_mat[idx1, idx2]

            ratio = (bl_after - bl_before) / bl_before

            if ratio > 0.2:
                print(
                    path_contcar, elem1, idx1, bl_before,
                    elem2, idx2, bl_after, ratio)
                break


main()
