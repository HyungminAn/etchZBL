import sys
import ase.io
from ase.geometry import get_distances
from multiprocessing import Pool, cpu_count
import numpy as np
from graph_tool import Graph
from graph_tool.topology import label_components
import cProfile
"""
This will delete designated molecules.

Original version : Changho Hong
modified 2023. 11. 02 by Hyungmin An
"""


def get_distances_all(
        R, selected_idx, indices,
        cell=None, pbc=None, mic=False, vector=False):
    """Return distances of atom No.i with a list of atoms.

    Use mic=True to use the Minimum Image Convention.
    vector=True gives the distance vector (from a to self[indices]).

    originally from ase.geometry.get_distances
    """
    p1 = [R[selected_idx]]
    p2 = R[indices]

    D, D_len = get_distances(p1, p2, cell=cell, pbc=pbc)

    if vector:
        D.shape = (-1, 3)
        return D
    else:
        D_len.shape = (-1,)
        return D_len


def find_nearest_neighbors(i, n_atoms, elem_idx, bl_mat, R, cell):
    '''
    Find nearest neighbors for atom i within the cutoff_distance.
    '''
    indices = np.arange(n_atoms)
    distances = get_distances_all(
        R, i, indices, cell=cell, pbc=True, mic=True)
    neighbors_logical = np.array([
        distances[j] < bl_mat[elem_idx-1, elem_idx-1]
        for j in indices
    ])
    neighbors_logical[i] = False
    neighbors = np.where(neighbors_logical)

    return (i, neighbors)


class atom_image():
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
        self.bond_length = self.generate_bondlength_matrix()
        self.deleting_molecules = self.generate_deleting_molecules()

    def generate_bondlength_matrix(self):
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

    def generate_deleting_molecules(self):
        '''
        (#Si, #O, #C, #H, #F)
        '''
        SiF2 = (1, 0, 0, 0, 2)
        SiF4 = (1, 0, 0, 0, 4)

        O2 = (0, 2, 0, 0, 0)
        F2 = (0, 0, 0, 0, 2)
        H2 = (0, 0, 0, 2, 0)
        CO = (0, 1, 1, 0, 0)

        CF3 = (0, 0, 1, 0, 3)

        CH2O = (0, 1, 1, 2, 0)
        CHFO = (0, 1, 1, 1, 1)
        CO2 = (0, 2, 1, 0, 0)
        COF = (0, 1, 1, 0, 1)
        COF2 = (0, 1, 1, 0, 2)
        COH = (0, 1, 1, 1, 0)
        H2O = (0, 1, 0, 2, 0)
        OF2 = (0, 1, 0, 0, 2)
        OHF = (0, 1, 0, 1, 1)

        return np.array((
            SiF2, SiF4, O2, F2, H2, CO, CF3,
            CH2O, CHFO, CO2, COF, COF2, COH, H2O, OF2, OHF,
            ))

    def read_atom_image(self, image_path):
        self.image_path = image_path
        self.atoms = ase.io.read(
            image_path, format='lammps-data',
            index=0, style="atomic", sort_by_id=False)
        self.num_atoms = len(self.atoms)
        self.atomic_numbers = self.atoms.get_atomic_numbers()
        self.n_elements = 5

    def find_NN(self):
        '''
        Create a multiprocessing Pool,
            and run the find_nearest_neighbors function for each atom.
        '''
        n_atoms = self.num_atoms
        arg_1 = [i for i in range(n_atoms)]
        arg_2 = [n_atoms for i in range(n_atoms)]
        arg_3 = self.atomic_numbers
        bl_mat = self.bond_length
        arg_4 = [bl_mat.copy() for i in range(n_atoms)]
        pos = self.atoms.get_positions()
        arg_5 = [pos.copy() for i in range(n_atoms)]
        cell = self.atoms.get_cell()
        arg_6 = [cell.copy() for i in range(n_atoms)]

        pool = Pool(cpu_count())
        self.nearest_neighbor = pool.starmap(
            find_nearest_neighbors,
            zip(arg_1, arg_2, arg_3, arg_4, arg_5, arg_6),
            )

    def draw_graph(self):
        self.graph = Graph(directed=False)
        self.graph.add_vertex(self.num_atoms)

        for (idx, neighbors) in self.nearest_neighbor:
            if neighbors[0].size == 0:
                continue

            for j in neighbors[0]:
                self.graph.add_edge(idx, j)

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
                stoichiometry[self.atomic_numbers[j]-1] += 1

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
                    stoichiometry[self.atomic_numbers[atom]-1] += 1
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
                for j in i:
                    self.to_delete_idx[n] = j
                    n += 1
        else:
            self.delete_before_write = 0

    def write_final_image(self):
        if self.delete_before_write == 1:
            del self.atoms[self.to_delete_idx]

        image_path_new = self.image_path.replace(".coo", "_after_removing.coo")
        ase.io.write(
            image_path_new, self.atoms,
            format='lammps-data', atom_style='atomic',
            velocities=True, specorder=["H", "He", "Li", "Be", "B"])


def main():
    Image = atom_image()
    Image.read_atom_image("CHF_shoot_180.coo")
    Image.find_NN()

    Image.get_delete_atoms("180")
    Image.write_final_image()


cProfile.run("main()", "result.prof")
