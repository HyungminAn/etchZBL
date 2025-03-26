import pickle

import numpy as np
from scipy.spatial import cKDTree
from utils import timeit
from utils import read_structure as read

class BondingData:
    """
    This class demonstrates how to collect neighbor information for carbon atoms,
    using a pre-built adjacency list (faster queries).

    Attributes:
        atoms (ase.Atoms): The ASE Atoms object.
        adjacency (list of sets): adjacency[i] is the set of neighbors for atom i.
        carbon_indices (list of int): Indices of all carbon atoms in 'atoms'.
        carbon_neighbors_info (list of dict): Detailed neighbor info for each carbon.
    """
    def __init__(self, atoms, adjacency):
        """
        Initialize the BondingData by:
          1. Storing the atoms and adjacency.
          2. Finding all carbon atoms (symbol == 'C').
          3. Collecting neighbor information for each carbon atom (using adjacency).
        """
        self.atoms = atoms
        self.adjacency = adjacency

        # Find all carbon indices
        self.carbon_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'C']

        # List of neighbor info for each carbon
        self.carbon_neighbors_info = []
        self._collect_carbon_neighbors()

    def _collect_carbon_neighbors(self):
        """
        For each carbon atom, build a dictionary of neighbor information:
          - carbon_index
          - num_neighbors
          - neighbor_indices
          - neighbor_symbols
          - neighbor_distances
        and store it in self.carbon_neighbors_info.
        """
        positions = self.atoms.get_positions()
        for c_idx in self.carbon_indices:
            # adjacency[c_idx] gives the set of neighbor indices
            neighbor_indices = list(self.adjacency[c_idx])
            neighbor_symbols = [self.atoms[i].symbol for i in neighbor_indices]

            # Calculate distances from carbon to each neighbor
            # c_pos = positions[c_idx]
            # neighbor_positions = positions[neighbor_indices]
            # dist_vecs = neighbor_positions - c_pos
            # distances = np.linalg.norm(dist_vecs, axis=1)

            info = {
                "carbon_index": c_idx,
                # "num_neighbors": len(neighbor_indices),
                # "neighbor_indices": neighbor_indices,
                "neighbor_symbols": neighbor_symbols,
                # "neighbor_distances": distances.tolist(),
            }
            self.carbon_neighbors_info.append(info)

        self.atoms = None  # Don't store the atoms in memory
        self.adjacency = None  # Don't store the adjacency in memory
        self.carbon_indices = None  # Don't store the carbon indices in memory


class BondDataGenerator:
    def __init__(self, path_to_cutoff_matrix):
        self.cutoff_matrix = self.read_cutoff_matrix(path_to_cutoff_matrix)

    def run(self, path_to_input_structure):
        atoms = read(path_to_input_structure)
        adjacency = self.build_adjacency(atoms, self.cutoff_matrix)
        return BondingData(atoms, adjacency)

    @timeit
    def read_cutoff_matrix(self, path_to_pickle):
        """
        Reads a 5x5 cutoff matrix from a pickle file.
        Each index [0..4] corresponds to [Si, O, C, H, F] by default.
        Adjust if your system differs.
        """
        with open(path_to_pickle, 'rb') as f:
            cutoff_matrix = pickle.load(f)
        return cutoff_matrix

    @timeit
    def build_adjacency(self, atoms, cutoff_matrix):
        atoms.wrap()
        positions = atoms.get_positions()
        positions[positions < -1e-10] = 0.0
        element_order = ["Si", "O", "C", "H", "F"]
        symbol_to_index = {sym: i for i, sym in enumerate(element_order)}
        all_symbols = atoms.get_chemical_symbols()
        atom_types = np.array([symbol_to_index[s] for s in all_symbols])
        cell_lengths = atoms.get_cell().lengths()
        try:
            tree = cKDTree(positions, boxsize=cell_lengths)
        except:
            breakpoint()
        max_cutoff = np.max(cutoff_matrix)
        pairs = tree.sparse_distance_matrix(tree, max_cutoff, output_type='coo_matrix')
        num_atoms = len(atoms)
        adjacency = [set() for _ in range(num_atoms)]
        row, col, dist = pairs.row, pairs.col, pairs.data
        valid = dist < cutoff_matrix[atom_types[row], atom_types[col]]
        for i, j in zip(row[valid], col[valid]):
            if i != j:
                adjacency[i].add(j)
                adjacency[j].add(i)
        return adjacency
