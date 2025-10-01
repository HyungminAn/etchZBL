import os
import re
import copy
import time
import pickle
from functools import wraps
from dataclasses import dataclass

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from ase.io import read
from ase.geometry import get_distances

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper


@dataclass
class PARAMS:
    ELMLIST = ['Si', 'O', 'C', 'H', 'F']

    ATOM_IDX_Si = 1
    ATOM_IDX_O = 2
    ATOM_IDX_C = 3
    ATOM_IDX_H = 4
    ATOM_IDX_F = 5

    ATOM_NUM_Si = 14
    ATOM_NUM_O = 8
    ATOM_NUM_C = 6
    ATOM_NUM_H = 1
    ATOM_NUM_F = 9

    LAMMPS_READ_OPTS = {
        "format": 'lammps-data',
        'sort_by_id': False,
        "Z_of_type": {
            ATOM_IDX_Si: ATOM_NUM_Si,
            ATOM_IDX_O: ATOM_NUM_O,
            ATOM_IDX_C: ATOM_NUM_C,
            ATOM_IDX_H: ATOM_NUM_H,
            ATOM_IDX_F: ATOM_NUM_F
        },
    }

    BYPRODUCT_LIST = [
        'SiF4', 'SiHF3', 'SiH2F2', 'SiH3F', 'SiH4', 'SiF2',
        'O2', 'H2', 'F2', 'CO', 'HF',
        'CF4', 'CHF3', 'CH2F2', 'CH3F', 'CH4', 'CF2',
        'H2O', 'OF2', 'OHF' 'CH2O', 'CHFO', 'COF2', 'CO2',
    ]

class DistMatrix:
    def __init__(self):
        self.matrix = self.gen_matrix()

    def gen_matrix(self):
        bond_length = np.array([
            [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
            [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
            [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
            [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
            [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
        ])
        return bond_length + bond_length.T - np.diagflat(np.diagonal(bond_length))


class ByProductRemover:
    def __init__(self, elmlist : list, byproduct : list):
        self.elmlist = elmlist
        self.byproduct_list = byproduct
        self.byproduct_dict  = {}
        self.byproduct_label = {}
        self._parse_byproducts()

    @timeit
    def run(self, structure, distance_matrix) -> None:
        """Remove byproducts of the calculation
        """
        ## Generate graph networks
        G_in, cluster_in = GraphBuilder.graph_call_in(structure, distance_matrix)
        removed_result = self._remove_byproduct(structure, cluster_in, fix_height=6.0)
        return removed_result

    @staticmethod
    def _key_to_label(chemical_symbols : list, number_of_atom : list) -> str:
        '''
        Convert the chemical symbols and number of atoms to the string label
        '''
        ## Sort the chemical symbols and number of atoms to the string label
        sorted_label = sorted(zip(chemical_symbols, number_of_atom), key = lambda x: x[0])
        label = ''.join([f'{symbol}{num}' for symbol, num in sorted_label])
        return label

    def _parse_byproducts(self) -> None:
        # Define the regular expression pattern
        epattern = '|'.join(self.elmlist)
        pattern = re.compile(r'(?P<element>'+epattern+r')(?P<count>\d*)')
        for  molecule  in self.byproduct_list:
            matches = pattern.findall(molecule)
            odict = {element: int(count) if count else 1 for element, count in matches}
            # Change dict to chemical symbols and number of atoms to the string label
            label = self._key_to_label(odict.keys(), odict.values())

            self.byproduct_dict[molecule] = odict
            self.byproduct_label[label] = molecule

    def _remove_byproduct(self, atoms, cluster_list : list, fix_height):
        oatom = copy.deepcopy(atoms)
        slab_idx = np.argmax([len(c) for c in cluster_list])
        pos_z = atoms.get_positions()[:, 2].flatten()
        h_slab_max = np.max(pos_z[np.array(list(cluster_list[slab_idx]))])

        result = []
        for idx, cluster in enumerate(cluster_list):
            ## Pass the cluster is slab
            if idx == slab_idx:
                continue

            h_cluster_min = np.min(pos_z[np.array(list(cluster))])
            h_cluster_avg = np.mean(pos_z[np.array(list(cluster))])
            is_within_fixed_region = h_cluster_min < fix_height
            if is_within_fixed_region:
                continue

            label, is_in_byproduct = self._is_in_byproduct(cluster, atoms)
            if is_in_byproduct:
                print(f'Cluster {idx} : {label} {is_in_byproduct}; {h_cluster_avg:.2f} {h_slab_max:.2f}')
                result.append((label, h_cluster_avg, h_slab_max, h_cluster_avg - h_slab_max))
                continue

        return result

    def _is_in_byproduct(self, cluster, atoms):
        '''
        Check the byproduct in the cluster is the one should be removed
        '''
        ndict = {}
        for cidx in cluster:
            symbol = atoms[cidx].symbol
            if symbol in ndict:
                ndict[symbol] += 1
            else:
                ndict[symbol] = 1

        label = self._key_to_label(ndict.keys(), ndict.values())
        return label, label in self.byproduct_label

class GraphBuilder:
    @staticmethod
    def graph_call_in(structure, distance_matrix):
        edges = GraphBuilder.find_edges(structure, distance_matrix)

        G_in = nx.Graph()
        G_in.add_edges_from(edges)
        clusters_in = list(nx.connected_components(G_in))
        return G_in, clusters_in

    @staticmethod
    def _split_into_bins(pos, cell, dist_matrix):
        # Define bin sizes for each axis based on cutoff distances
        slice_crit_x = slice_crit_y = slice_crit_z = np.max(dist_matrix.flatten())
        cell_x, cell_y, _ = cell.diagonal()

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

    @staticmethod
    def find_edges(image, dist_matrix):
        pos = image.get_positions()
        cell = image.get_cell()
        bins, bin_dict = GraphBuilder._split_into_bins(pos, cell, dist_matrix)

        atom_types = image.get_array('type')-1

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
            cutoff_distances = dist_matrix[atom_types_current][:, atom_types_neighboring]

            # Identify pairs within the cutoff distance
            row, col = np.where(D_len < cutoff_distances)

            # Map local indices back to global indices and add to all_edges
            for r, c in zip(row, col):
                all_edges.append((current_bin_indices[r], neighboring_bin_indices[c]))

        return all_edges


@timeit
def get_data(src_list, n_structure, path_save="data.pkl"):
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            return pickle.load(f)

    elmlist = PARAMS.ELMLIST
    distance_matrix = DistMatrix().matrix
    byproduct_list = PARAMS.BYPRODUCT_LIST
    Runner = ByProductRemover(elmlist, byproduct_list)

    result = {}
    for i in range(0, n_structure):
        filename = None
        for src in src_list:
            filename = f"{src}/str_shoot_{i}.coo"
            if os.path.exists(filename):
                break
        if filename is None:
            print(f"File not found for index {i}. Skipping...")
            continue
        print(f"Processing {filename}")
        structure = read(filename, **PARAMS.LAMMPS_READ_OPTS)
        result[i] = Runner.run(structure, distance_matrix)

    with open(path_save, 'wb') as f:
        pickle.dump(result, f)

    return result


@timeit
def plot(data):
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    removal_heights = []
    for key, my_list in data.items():
        for (label, h_cluster_avg, h_slab_max, h_diff) in my_list:
            removal_heights.append(h_diff)
    removal_heights = np.array(removal_heights) / 10  # Convert to nm
    ax.hist(removal_heights, bins=100, orientation='horizontal', alpha=0.8)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Count')
    ax.set_ylabel(r'h$_{\mathrm{removal}}-$h$_{\mathrm{slab}}$ (nm)')
    ax.set_title(f'Removal height distribution')
    ax.text(0.98, 0.98, f"Total count: {len(removal_heights)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10, color='black')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig('removal_height.png', dpi=200)


def main():
    src_list = [
        "/data_etch/data_HM/nurion/set_3/CF_300_coo/CF/300eV",
        "/data2/andynn/ZBL_modify/250331_ContinueRun/ContinueRun/CF_300",
        ]
    n_structure = 15000
    data = get_data(src_list, n_structure)
    plot(data)


if __name__ == "__main__":
    main()
