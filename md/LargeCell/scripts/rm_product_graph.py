import sys

import numpy as np

from ase.io import write
from graph_tool.topology import label_components
from AtomImage import AtomImage
"""
This will delete designated molecules.

Original version : Changho Hong
modified 2023. 11. 02 by Hyungmin An
"""


def generate_bondlength_matrix():
    '''
    In lammps idx, Si:1 O:2, C:3, H:4, F:5

    Current : 1.3 * (equilibrium bond length)
        * too small value will erase much molecules
    '''
    bond_length = np.array([
        [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
        [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
        [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
        [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
        [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
    ])
    n_rows, n_cols = bond_length.shape
    for row in range(n_rows):
        for col in range(row):
            bond_length[row, col] = \
                bond_length[col, row]
    return bond_length


def generate_deleting_molecules():
    '''
    (#Si, #O, #C, #H, #F)
    '''
    gas_to_remove = {
        "SiF2": (1, 0, 0, 0, 2),
        "SiF4": (1, 0, 0, 0, 4),

        "O2": (0, 2, 0, 0, 0),
        "F2": (0, 0, 0, 0, 2),
        "H2": (0, 0, 0, 2, 0),
        "CO": (0, 1, 1, 0, 0),
        "HF": (0, 0, 1, 0, 1),

        "CF": (0, 0, 1, 0, 1),
        "CH": (0, 0, 1, 1, 0),
        "CH2": (0, 0, 1, 2, 0),
        "CHF": (0, 0, 1, 1, 1),
        "CF2": (0, 0, 1, 0, 2),
        "CH3": (0, 0, 1, 3, 0),
        "CH2F": (0, 0, 1, 2, 1),
        "CHF2": (0, 0, 1, 1, 2),
        "CF3": (0, 0, 1, 0, 3),
        "CH4": (0, 0, 1, 4, 0),
        "CH3F": (0, 0, 1, 3, 1),
        "CH2F2": (0, 0, 1, 2, 2),
        "CHF3": (0, 0, 1, 1, 3),
        "CF4": (0, 0, 1, 0, 4),

        "CH2O": (0, 1, 1, 2, 0),
        "CHFO": (0, 1, 1, 1, 1),
        "COF2": (0, 1, 1, 0, 2),
        "CO2": (0, 2, 1, 0, 0),
        "H2O": (0, 1, 0, 2, 0),
        "OF2": (0, 1, 0, 0, 2),
        "OHF": (0, 1, 0, 1, 1),
    }

    return np.array([i for i in gas_to_remove.values()])


class DeleteAtomImage(AtomImage):
    def __init__(self):
        self.bond_length = generate_bondlength_matrix()
        self.deleting_molecules = generate_deleting_molecules()

    def get_tot_N_deleting(self):
        self.draw_graph()
        cluster, hist = label_components(self.graph)
        slab_idx = np.argmax(hist)
        cluster_idx = [i for i in range(len(hist))]
        cluster_idx.pop(slab_idx)

        to_delete_list = []
        tot_N_deleting = 0

        for i in cluster_idx:
            atom_in_cluster_idx = np.argwhere(cluster.a == i)
            stoichiometry = np.zeros(self.n_elements, dtype=int)

            for j in atom_in_cluster_idx:
                stoichiometry[self.atomic_numbers[j]] += 1

            is_to_be_deleted = np.any(
                np.all(self.deleting_molecules == stoichiometry, axis=1))
            if not is_to_be_deleted:
                continue

            tot_N_deleting += np.sum(stoichiometry)
            to_delete_list.append(atom_in_cluster_idx)

        return tot_N_deleting, to_delete_list

    def write_delete_log(self, to_delete_list, current_incidence):
        with open("delete.log", 'a') as log:
            w = log.write
            for mol_to_delete in to_delete_list:
                stoichiometry = np.zeros(self.n_elements, dtype=int)
                for atom in mol_to_delete:
                    stoichiometry[self.atomic_numbers[atom]] += 1
                line = ", ".join(map(str, stoichiometry))
                w(f"Current_incidence: {current_incidence} / ")
                w(f"stoichiometry: {line} / ")
                line = ", ".join([str(atom[0]) for atom in mol_to_delete])
                w(f"del_atom_idx: {line}\n")

    def get_delete_atoms(self, current_incidence):
        tot_N_deleting, to_delete_list = self.get_tot_N_deleting()
        cond = tot_N_deleting > 0
        if cond:
            self.delete_before_write = 1
            self.write_delete_log(to_delete_list, current_incidence)
            n = 0
            self.to_delete_idx = np.zeros(tot_N_deleting, dtype=int)
            for i in to_delete_list:
                i = i.flatten()
                for j in i:
                    self.to_delete_idx[n] = j
                    n += 1
        else:
            self.delete_before_write = 0

    def write_final_image(self):
        if self.delete_before_write == 1:
            del self.atoms[self.to_delete_idx]

        image_path_new = self.image_path.replace(".coo", "_after_removing.coo")
        write(
            image_path_new, self.atoms,
            format='lammps-data', atom_style='atomic',
            velocities=True, specorder=['Si', 'O', 'C', 'H', 'F'])


def main():
    path_image = sys.argv[1]
    current_incidence = sys.argv[2]

    Image = DeleteAtomImage()
    Image.read_atom_image(path_image)
    Image.find_NN()

    Image.get_delete_atoms(current_incidence)
    Image.write_final_image()


if __name__ == '__main__':
    main()
