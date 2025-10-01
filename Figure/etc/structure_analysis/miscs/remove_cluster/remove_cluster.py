import sys
import torch
from ase.io import read
import networkx as nx
import sys
from dataclasses import dataclass

from ase.io import read
from ase.geometry import get_distances
import numpy as np

import time
from functools import wraps
import cProfile
import pickle

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds")
        return result
    return wrapper

@dataclass
class READ_OPTS:
    IDX_Si, ATOM_NUM_Si, ATOM_IDX_Si = 0, 14, 1
    IDX_O, ATOM_NUM_O, ATOM_IDX_O = 1, 8, 2
    IDX_C, ATOM_NUM_C, ATOM_IDX_C = 2, 6, 3
    IDX_H, ATOM_NUM_H, ATOM_IDX_H = 3, 1, 4
    IDX_F, ATOM_NUM_F, ATOM_IDX_F = 4, 9, 5

    LAMMPS_DATA_OPTS = {
            'format': 'lammps-data',
            'Z_of_type': {
                int(ATOM_IDX_Si): ATOM_NUM_Si,
                int(ATOM_IDX_O): ATOM_NUM_O,
                int(ATOM_IDX_C): ATOM_NUM_C,
                int(ATOM_IDX_H): ATOM_NUM_H,
                int(ATOM_IDX_F): ATOM_NUM_F,
                },
            }
    ATOM_NUM_MATCHER = {
            ATOM_NUM_Si: IDX_Si,
            ATOM_NUM_O: IDX_O,
            ATOM_NUM_C: IDX_C,
            ATOM_NUM_H: IDX_H,
            ATOM_NUM_F: IDX_F,
            }
    ATOM_TYPE_DICT = {
            'Si': IDX_Si,
            'O': IDX_O,
            'C': IDX_C,
            'H': IDX_H,
            'F': IDX_F,
            }

class DistMatrix:
    def __init__(self, method=None):
        if method == 'torch':
            self.matrix = self.gen_matrix_torch()
        elif method == 'numpy':
            self.matrix = self.gen_matrix_np()
        else:
            raise ValueError('Invalid method')

    def gen_matrix_torch(self):
        arr = torch.tensor([
            [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
            [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
            [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
            [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
            [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
        ])
        return arr + arr.t() - torch.diag(arr.diag())

    def gen_matrix_np(self):
        arr = np.array([
            [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
            [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
            [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
            [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
            [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
        ])
        arr_diag = np.zeros_like(arr)
        np.fill_diagonal(arr_diag, np.diag(arr))
        return arr + arr.T - arr_diag


class GraphBuilder:
    @staticmethod
    @measure_time
    def run(structure, device, distance_matrix) -> None:
        edges = GraphBuilder.find_edges(structure, device, distance_matrix)
        graph = nx.Graph()
        graph.add_edges_from(edges)
        clusters = list(nx.connected_components(graph))
        return graph, clusters

    @staticmethod
    def find_edges(structure, device, distance_matrix) -> list:
        atomic_n = torch.tensor([READ_OPTS.ATOM_NUM_MATCHER[i] for i in
                                 structure.get_atomic_numbers()]).to(device)
        distance = torch.tensor(structure.get_all_distances(mic=True)).to(device)
        distance.fill_diagonal_(float('inf'))
        distance_matrix = distance_matrix.to(device)
        cutoff = distance_matrix[atomic_n][:, atomic_n]
        is_connected = distance < cutoff
        edges = torch.nonzero(is_connected, as_tuple=False).tolist()
        return edges


class SpatialDecompositionGraphBuilder:
    def __init__(self, path_image, dist_matrix):
        self.path_image = path_image
        self.image = read(path_image, **READ_OPTS.LAMMPS_DATA_OPTS)
        self.dist_matrix = dist_matrix

    @measure_time
    def run(self):
        edges = self.find_edges()
        graph = nx.Graph()
        graph.add_edges_from(edges)
        clusters = list(nx.connected_components(graph))
        return graph, clusters

    def find_edges(self):
        pos = self.image.get_positions()
        bins, bin_dict = self._split_into_bins(pos)

        cell = self.image.get_cell()
        symbols = self.image.get_chemical_symbols()
        atom_types = np.array([READ_OPTS.ATOM_TYPE_DICT[s] for s in symbols])

        # Initialize a list to store connectivity edges
        all_edges = []

        # Iterate over bins and calculate connectivity within and between neighboring bins
        for current_bin in range(1, len(bins) + 1):
            print(current_bin)
            # Get the indices of atoms in the current bin
            current_bin_indices = bin_dict[current_bin]

            # Get the indices of atoms in the neighboring bins
            neighboring_bins = [current_bin, current_bin + 1]
            neighboring_bins = [b for b in neighboring_bins if b in bin_dict]
            neighboring_bin_indices = [
                atom_idx for b in neighboring_bins for atom_idx in bin_dict[b]
            ]

            # Calculate pairwise distances between atoms in the current and neighboring bins
            if not neighboring_bin_indices:
                continue

            pos_current = pos[current_bin_indices]
            pos_neighboring = pos[neighboring_bin_indices]

            # Calculate distances with minimum image convention (mic=True)
            _, D_len = get_distances(pos_current, p2=pos_neighboring, cell=cell, pbc=True)

            # Calculate cutoff distances for each pair
            atom_types_current = atom_types[current_bin_indices]
            atom_types_neighboring = atom_types[neighboring_bin_indices]
            cutoff_distances = self.dist_matrix[atom_types_current][:, atom_types_neighboring]

            # Identify pairs within the cutoff distance
            row, col = np.where(D_len < cutoff_distances)

            # Map local indices back to global indices and add to all_edges
            for r, c in zip(row, col):
                all_edges.append((current_bin_indices[r], neighboring_bin_indices[c]))

        return all_edges

    def _split_into_bins(self, pos):
        slice_crit_z = np.max(self.dist_matrix.flatten())
        max_val = np.max(pos[:, 2])
        bins = np.arange(0, max_val + slice_crit_z, slice_crit_z)
        bin_indices = np.digitize(pos[:, 2], bins)

        # Create a dictionary to store atom indices in each bin
        bin_dict = {i: [] for i in range(1, len(bins) + 1)}
        for i, bin_idx in enumerate(bin_indices):
            bin_dict[bin_idx].append(i)
        return bins, bin_dict

class SpatialDecompositionGraphBuilder3D:
    def __init__(self, path_image, dist_matrix):
        self.path_image = path_image
        self.image = read(path_image, **READ_OPTS.LAMMPS_DATA_OPTS)
        self.dist_matrix = dist_matrix

    @measure_time
    def run(self):
        edges = self.find_edges()
        graph = nx.Graph()
        graph.add_edges_from(edges)
        clusters = list(nx.connected_components(graph))
        return graph, clusters

    def _split_into_bins(self, pos, cell):
        # Define bin sizes for each axis based on cutoff distances
        slice_crit_x = slice_crit_y = slice_crit_z = np.max(self.dist_matrix.flatten())
        cell_x, cell_y, cell_z = cell.diagonal()

        # Determine bins for each axis
        x_bins = np.arange(0, cell_x, slice_crit_x)[:-1]
        y_bins = np.arange(0, cell_y, slice_crit_y)[:-1]
        z_bins = np.arange(0, np.max(pos[:, 2]) + slice_crit_z, slice_crit_z)

        # Digitize positions into bins
        x_indices = np.digitize(pos[:, 0], x_bins)
        y_indices = np.digitize(pos[:, 1], y_bins)
        z_indices = np.digitize(pos[:, 2], z_bins)

        # Combine x, y, z indices into unique 3D bin keys
        bin_keys = list(zip(x_indices, y_indices, z_indices))

        # Create a dictionary mapping each bin to atom indices
        bin_dict = {}
        for i, key in enumerate(bin_keys):
            if key not in bin_dict:
                bin_dict[key] = []
            bin_dict[key].append(i)
        return (x_bins, y_bins, z_bins), bin_dict

    @measure_time
    def find_edges(self):
        pos = self.image.get_positions()
        cell = self.image.get_cell()
        bins, bin_dict = self._split_into_bins(pos, cell)

        symbols = self.image.get_chemical_symbols()
        atom_types = np.array([READ_OPTS.ATOM_TYPE_DICT[s] for s in symbols])

        # Initialize a list to store connectivity edges
        all_edges = []

        # Iterate over bins and calculate connectivity within and between neighboring bins
        x_bins, y_bins, z_bins = bins
        for current_bin in bin_dict:
            current_bin_indices = bin_dict[current_bin]

            # Get neighboring bins (3D neighborhood)
            neighboring_bins = [
                (
                    ((current_bin[0] + dx - 1) % len(x_bins)) + 1,
                    ((current_bin[1] + dy - 1) % len(y_bins)) + 1,
                    ((current_bin[2] + dz - 1) % len(z_bins)) + 1
                )
                for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
            ]
            neighboring_bins = [b for b in neighboring_bins if b in bin_dict]

            # Get atom indices for neighboring bins
            neighboring_bin_indices = [
                atom_idx for b in neighboring_bins for atom_idx in bin_dict[b]
            ]

            # Calculate pairwise distances between atoms in the current and neighboring bins
            if not neighboring_bin_indices:
                continue

            pos_current = pos[current_bin_indices]
            pos_neighboring = pos[neighboring_bin_indices]

            # Calculate distances with minimum image convention (mic=True)
            _, D_len = get_distances(pos_current, p2=pos_neighboring, cell=cell, pbc=True)

            # Calculate cutoff distances for each pair
            atom_types_current = atom_types[current_bin_indices]
            atom_types_neighboring = atom_types[neighboring_bin_indices]
            cutoff_distances = self.dist_matrix[atom_types_current][:, atom_types_neighboring]

            # Identify pairs within the cutoff distance
            row, col = np.where(D_len < cutoff_distances)

            # Map local indices back to global indices and add to all_edges
            for r, c in zip(row, col):
                all_edges.append((current_bin_indices[r], neighboring_bin_indices[c]))

        return all_edges


class ClusterRemover:
    @staticmethod
    def run(path_image, device, dist_matrix, method=None):
        image = read(path_image, **READ_OPTS.LAMMPS_DATA_OPTS)
        if method == 'original':
            graph, clusters = GraphBuilder.run(image, device, dist_matrix)
            path_save = 'clusters_original.pkl'
        elif method == 'spatial':
            graph, clusters = SpatialDecompositionGraphBuilder(path_image, dist_matrix).run()
            path_save = 'clusters_spatial.pkl'
        elif method == 'spatial3d':
            graph, clusters = SpatialDecompositionGraphBuilder3D(path_image, dist_matrix).run()
            path_save = 'clusters_spatial3d.pkl'

        else:
            raise ValueError('Invalid method')

        cluster_dict = {i: list(cluster) for i, cluster in enumerate(clusters)}
        with open(path_save, 'wb') as f:
            pickle.dump(cluster_dict, f)

    @staticmethod
    def compare_method():
        path_save_original = 'clusters_original.pkl'
        path_save_spatial = 'clusters_spatial3d.pkl'
        with open(path_save_original, 'rb') as f:
            cluster_dict_original = pickle.load(f)
        with open(path_save_spatial, 'rb') as f:
            cluster_dict_spatial = pickle.load(f)

        for key1 in cluster_dict_original:
            len_1 = len(cluster_dict_original[key1])
            for key2 in cluster_dict_spatial:
                len_2 = len(cluster_dict_spatial[key2])
                if set(cluster_dict_original[key1]) == set(cluster_dict_spatial[key2]):
                    print(f"Cluster {key1} (len: {len_1}) is the same as Cluster {key2} (len: {len_2})")
                else:
                    print(f"Cluster {key1} (len: {len_1}) differs to Cluster {key2} (len: {len_2})")


def main():
    path_image = sys.argv[1]
    device = 'cpu'

    # ClusterRemover.run(path_image, device, DistMatrix(method='torch').matrix, method='original')
    # ClusterRemover.run(path_image, device, DistMatrix(method='numpy').matrix, method='spatial')
    ClusterRemover.run(path_image, device, DistMatrix(method='numpy').matrix, method='spatial3d')

    # ClusterRemover.compare_method()


if __name__ == '__main__':
    cProfile.run('main()', 'result.prof')
