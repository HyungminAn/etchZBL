from multiprocessing import Pool, cpu_count

import numpy as np

from ase.io import read
from graph_tool import Graph
"""
This will delete designated molecules.

Original version : Changho Hong
modified 2023. 11. 02 by Hyungmin An
"""


def generate_bondlength_matrix():
    '''
    In lammps idx, Si:1 O:2, C:3, H:4, F:5

    Current : 1.15 * (equilibrium bond length)
    '''
    bond_length = np.array([
        [2.62476, 1.75847, 1.98436, 1.77338, 1.88066],
        [0.00000, 1.41798, 1.31503, 1.13510, 1.57798],
        [0.00000, 0.00000, 1.51158, 1.30897, 1.49227],
        [0.00000, 0.00000, 0.00000, 0.86282, 1.07879],
        [0.00000, 0.00000, 0.00000, 0.00000, 1.63796],
    ])
    n_rows, n_cols = bond_length.shape
    for row in range(n_rows):
        for col in range(row):
            bond_length[row, col] = \
                bond_length[col, row]
    return bond_length


class AtomImage():
    '''
    * member variables
        bond_length: N by N matrix; criteria for connectivity
            (N: number of elements)
        deleting_molecules: M by N matrix; information for molecules to delete
            (M: number of types of molecules to delete)

        image_path: path
        atoms: ASE image
        num_atoms: number of atoms in the image
        atomic_numbers: list of atomic numbers in the image

        nearest_neighbor: list of length `num_atoms`;
            list components are a tuple of (
                    idx_atom, [indices of neighbor atoms of `idx_atom`]
                    )
    '''
    def __init__(self):
        self.bond_length = generate_bondlength_matrix()

    def read_atom_image(self, image_path):
        self.image_path = image_path
        read_options = {
            'format': 'lammps-data',
            'index': 0,
            'atom_style': 'atomic',
            'sort_by_id': False,
            'Z_of_type': {
                1: 14,  # Si
                2: 8,   # O
                3: 6,   # C
                4: 1,   # H
                5: 9,   # F
                }
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

    def find_nearest_neighbors(self, i):
        '''
        Find nearest neighbors for atom i within the cutoff_distance.
        '''
        n_atoms = self.num_atoms
        elem_idx = self.atomic_numbers
        bl_mat = self.bond_length

        indices = np.arange(n_atoms)
        distances = self.atoms.get_distances(i, indices, mic=True)
        neighbors_logical = np.array([
            distances[j] < bl_mat[elem_idx[i], elem_idx[j]]
            for j in indices
        ])
        neighbors_logical[i] = False
        neighbors = np.where(neighbors_logical)

        return (i, neighbors)

    def find_NN(self):
        '''
        Create a multiprocessing Pool,
            and run the find_nearest_neighbors function for each atom.
        '''
        pool = Pool(cpu_count())
        self.nearest_neighbor = pool.starmap(
            self.find_nearest_neighbors,
            [(i, ) for i in range(self.num_atoms)])

    def draw_graph(self):
        self.graph = Graph(directed=False)
        self.graph.add_vertex(self.num_atoms)

        for (idx, neighbors) in self.nearest_neighbor:
            if neighbors[0].size == 0:
                continue

            for j in neighbors[0]:
                self.graph.add_edge(idx, j)
