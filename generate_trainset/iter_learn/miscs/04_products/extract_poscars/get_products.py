import os
import sys

import numpy as np

from ase.io import read, write
from ase.geometry import get_distances
from graph_tool.topology import label_components
from AtomImage import AtomImage
"""
This code extracts the molecules that are not connected to the slab.
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


class ExtractAtomImage(AtomImage):
    def __init__(self, image_idx):
        self.bond_length = generate_bondlength_matrix()
        self.image_idx = image_idx

    def read_atom_image(self, image_path):
        self.image_path = image_path
        read_options = {
            'format': 'lammps-dump-text',
            'index': -1,
            }
        self.atoms = read(image_path, **read_options)
        self.num_atoms = len(self.atoms)
        atomic_number_dict = {
            14: ('Si', 0),
            8: ('O', 1),
            6: ('C', 2),
            1: ('H', 3),
            9: ('F', 4),
            }
        self.atomic_numbers = np.array([
            atomic_number_dict[i][1]
            for i in self.atoms.get_atomic_numbers()
        ])

        self.n_elements = 5

    def get_idx_extract_atoms(self):
        self.draw_graph()
        cluster, hist = label_components(self.graph)
        slab_idx = np.argmax(hist)
        cluster_idx = [i for i in range(len(hist))]
        cluster_idx.pop(slab_idx)

        extract_list = []
        for i in cluster_idx:
            atom_in_cluster_idx = np.argwhere(cluster.a == i)
            extract_list.append(atom_in_cluster_idx)

        return extract_list

    def extract_atoms(self, label, exclude_list):
        idx_list = self.get_idx_extract_atoms()
        container = self.atoms.copy()
        while container:
            container.pop()
        cell = container.get_cell()
        cell_size = 15  # angstrom
        cell[0, 0] = cell[1, 1] = cell[2, 2] = cell_size
        container.set_cell(cell)
        cell_center = np.array([cell_size / 2] * 3)

        write_options = {
            'format': 'lammps-data',
            'atom_style': 'atomic',
            'velocities': True,
            'specorder': ['Si', 'O', 'C', 'H', 'F'],
        }

        for mol_count, idx_molecule in enumerate(idx_list):
            idx_molecule = idx_molecule.flatten()
            poscar = container.copy()
            for atom in self.atoms[idx_molecule]:
                poscar.append(atom)

            shift_molecule_to_cell_center(poscar, cell_center)
            formula = poscar.get_chemical_formula()

            if formula in exclude_list:
                continue

            name = f"poscars/POSCAR_{label}_{self.image_idx}_{mol_count}_{formula}"

            write(name, poscar, **write_options)
            print(f"Extracted {name}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python get_products.py [path_image] [label]")
        sys.exit(1)
    path_image = sys.argv[1]
    label = sys.argv[2]

    exclude_list = [
        'CF',
        'CF3',
        'CFH2',
    ]

    os.makedirs('poscars', exist_ok=True)

    n_images = len(read(path_image, index=':'))
    for image_idx in np.arange(0, n_images, 10):
        Image = ExtractAtomImage(image_idx)
        Image.read_atom_image(path_image)
        Image.find_NN()

        Image.extract_atoms(label, exclude_list)
        print(f"{path_image} {image_idx}/{n_images} Done")


if __name__ == '__main__':
    main()
