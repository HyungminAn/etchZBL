import sys
import torch
from ase.io import read
import networkx as nx
import sys
from dataclasses import dataclass

from ase.io import read
from ase.geometry import get_distances
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint as pp

import time
from functools import wraps
from ase.calculators.lammps import convert

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
class CONSTANTS:
    ANGPS_TO_MPS = 100
    AMU_TO_KG = 1.66053906660e-27
    J_TO_EV = 6.242e+18
    KB = 8.617333262145e-5  # eV/K

@dataclass
class MD_SETTINGS:
    FIX_HEIGHT = 6
    TEMP_HEIGHT = 6

@dataclass
class READ_OPTS:
    LAMMPS_DATA_OPTS = {
            'format': 'lammps-data',
            'Z_of_type': {
                1: 14,
                2: 8,
                3: 6,
                4: 1,
                5: 9,
                }
            }

class KineticEnergyMatrixBuilder:
    def __init__(self):
        pass

    @measure_time
    def get_data(self, image_1, image_2, src_thermo):
        timestep = self._read_timestep(src_thermo)
        id_to_idx_map_1 = self._make_id_to_idx_map(image_1)
        id_to_idx_map_2 = self._make_id_to_idx_map(image_2)
        xyz_1 = self._make_pos_matrix(image_1, id_to_idx_map_1)
        xyz_2 = self._make_pos_matrix(image_2, id_to_idx_map_2)

        cell = image_2.get_cell()
        disp_matrix = self._get_disp_matrix(xyz_1, xyz_2, cell)
        speed_matrix = disp_matrix / timestep
        mass_list = self._make_mass_list(image_2, id_to_idx_map_2)
        kE_matrix = self._make_kinetic_energy_matrix(speed_matrix, mass_list)

        return kE_matrix

    def _read_timestep(self, src_thermo):
        '''
        Read timestep from thermo file
        '''
        matrix = np.loadtxt(src_thermo, skiprows=2, usecols=(0, 1))
        _, unique_indices = np.unique(matrix[:, 0], return_index=True)
        reduced_matrix = matrix[unique_indices]
        timestep = reduced_matrix[-2, 1] - reduced_matrix[-1, 1]
        return timestep

    def _make_id_to_idx_map(self, image):
        id_set = set()
        id_set.update(image.get_array('id'))
        return { k: i for i, k in enumerate(id_set) }

    def _make_pos_matrix(self, image, id_to_idx_map):
        '''
        From dump, make position matrix which has shape of (len_atom, len_dump, 3)
        '''
        len_atom = len(id_to_idx_map)
        pos_matrix = np.zeros((len_atom, 3))
        atom_id = np.array([id_to_idx_map[i] for i in image.get_array('id')])
        pos_matrix[atom_id, :] = image.get_positions()
        return pos_matrix

    def _get_disp_matrix(self, xyz, xyz_prev, cell):
        '''
        Get displacement matrix from position matrix (selected atoms)
        '''
        _, D_len = get_distances(xyz, p2=xyz_prev, pbc=True, cell=cell)
        disp = np.diag(D_len)
        return disp

    def _make_kinetic_energy_matrix(self, speed_matrix, mass_list):
        '''
        Make temperature matrix from speed matrix and mass list
        '''
        len_atom = len(speed_matrix)
        kE = CONSTANTS.J_TO_EV * 0.5 * (mass_list * CONSTANTS.AMU_TO_KG) * (speed_matrix * CONSTANTS.ANGPS_TO_MPS) **2
        return kE

    def _make_mass_list(self, image, id_to_idx_map):
        len_atom = len(id_to_idx_map)
        mass_list = np.zeros(len_atom)
        mass = image.get_masses()
        atom_id = image.get_array('id')
        for i, m in zip(atom_id, mass):
            mass_list[id_to_idx_map[i]] = m
        return mass_list


class DistMatrix:
    def __init__(self):
        self.matrix = self.gen_matrix()

    def gen_matrix(self):
        bond_length = torch.tensor([
            [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
            [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
            [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
            [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
            [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
        ])
        return bond_length + bond_length.t() - torch.diag(bond_length.diag())


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
        '''
        Convert 0 value to infinite value in distance matrix
        Torch module to create networks accelerate the calculation
        Create graph networks which is the distance matrix is smaller than the cutoff

        Step 2: Generate edges from connectivity matrix
        Step 3: Create an undirected graph using NetworkX
        Step 4: Analyze connected components (clusters)
        '''
        atomic_number_matcher = {
                14: 0,
                8: 1,
                6: 2,
                1: 3,
                9: 4,
                }
        atomic_n = torch.tensor([atomic_number_matcher[i] for i in
                                 structure.get_atomic_numbers()]).to(device)
        distance = torch.tensor(structure.get_all_distances(mic=True)).to(device)
        distance.fill_diagonal_(float('inf'))

        # Ensure distance_matrix is on the same device as atomic_n
        distance_matrix = distance_matrix.to(device)

        cutoff = distance_matrix[atomic_n][:, atomic_n]
        # cutoff = (distance_matrix_at_n[:, None] + distance_matrix_at_n[None, :])
        is_connected = distance < cutoff
        edges = torch.nonzero(is_connected, as_tuple=False).tolist()  # Convert to list of pairs
        return edges


class ClusterTempAnalyzer:
    @staticmethod
    def run(path_image, path_thermo, device, dist_matrix):
        image = read(path_image, **READ_OPTS.LAMMPS_DATA_OPTS)
        ANGSTROM_PS_TO_MPS = 1e+2
        VEL_CONV_FACTOR = convert(1.0, 'velocity', 'ASE', 'metal') * ANGSTROM_PS_TO_MPS
        velocities = image.get_velocities() * VEL_CONV_FACTOR
        graph, clusters = GraphBuilder.run(image, device, dist_matrix)
        clusters = ClusterTempAnalyzer.filter_by_height(clusters, image, MD_SETTINGS.FIX_HEIGHT)
        cluster_temp = ClusterTempAnalyzer._get_cluster_temp(kE_matrix, clusters)

        # str_1, str_2 = read(path_dump, format='lammps-dump-text', index='-2:')
        # kE_matrix = KineticEnergyMatrixBuilder().get_data(str_1, str_2, path_thermo)

        # for n_atoms, temp, cluster in cluster_temp:
        #     print(f"{n_atoms} atoms: {temp:.2f} K")
        #     print(cluster)

    # @staticmethod
    # def _get_cluster_temp(kE_matrix, clusters):
    #     cluster_temp = []
    #     for cluster in clusters:
    #         cluster = np.array(list(cluster))
    #         n_atoms = len(cluster)
    #         n_dof = 3 * n_atoms
    #         temp = 2 * kE_matrix[cluster].sum() / (n_dof * CONSTANTS.KB)
    #         cluster_temp.append((n_atoms, temp, cluster))
    #     return cluster_temp

    # @staticmethod
    # def filter_by_height(clusters, structure, height):
    #     pos = structure.get_positions()
    #     mask = set([i for i, p in enumerate(pos) if p[2] > height])
    #     clusters = [cluster & mask for cluster in clusters]
    #     return clusters

def main():
    path_image = sys.argv[1]
    # path_dump = sys.argv[1]
    path_thermo = sys.argv[2]
    dist_matrix = DistMatrix().matrix
    device = 'cpu'

    ClusterTempAnalyzer.run(path_image, path_thermo, device, dist_matrix)


if __name__ == '__main__':
    main()
