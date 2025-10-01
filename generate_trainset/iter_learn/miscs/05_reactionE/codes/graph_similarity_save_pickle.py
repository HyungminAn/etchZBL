import sys
import numpy as np
from ase.neighborlist import neighbor_list
from ase.io import read
import pickle
import graph_tool.all as gt
"""
Original code by Changho Hong

This code reads dump file,
and finds the stable configurations,
using graph similiarity.
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
            bond_length[row, col] = bond_length[col, row]

    return bond_length


def generate_bondlength_dict(bond_length_mat):
    n_species, _ = bond_length_mat.shape
    atom_idx = {
        0: 'Si',
        1: 'O',
        2: 'C',
        3: 'H',
        4: 'F',
    }
    bl_dict = {}
    for i in range(n_species):
        for j in range(i, n_species):
            key = (atom_idx[i], atom_idx[j])
            value = bond_length_mat[i, j]
            bl_dict[key] = value
    return bl_dict


class atom_image():
    def __init__(self):
        self.bond_length = generate_bondlength_matrix()
        self.bond_length_dict = generate_bondlength_dict(self.bond_length)

    def set_image_path(self,image_path):
        self.image_path=image_path

    def read_dump(self):
        self.dump = read(self.image_path,format='lammps-dump-text',index=":")

    def get_graph(self):
        n_images = len(self.dump)
        self.graphs = [gt.Graph(directed = False) for _ in range(n_images)]
        self.vprops = [[] for _ in range(n_images)]

        for i in range(n_images):
            NN_list = neighbor_list('ij',self.dump[i],self.bond_length_dict)
            self.vprops[i] = self.graphs[i].new_vertex_property("short",vals=self.dump[i].get_atomic_numbers())
            self.graphs[i].add_edge_list(zip(NN_list[0],NN_list[1]))


def is_two_graph_same(graph1, graph2):
    return gt.similarity(graph1, graph2, norm=False, distance=True, asymmetric=True) == 0.0


def is_stable_enough(match_idx, i, sampling_last, criteria=0.9):
    return np.sum(match_idx[i-sampling_last:i]==match_idx[i])/sampling_last >= criteria


def get_stable_config(Image, criteria=0.9, sampling_last=100):
    unique_snapshot_idx=[1]
    n_images = len(Image.dump)
    match_idx = np.zeros(n_images-1, dtype=int)

    for i in range(1, n_images):
        for j in unique_snapshot_idx:
            if is_two_graph_same(Image.graphs[i], Image.graphs[j]):
                match_idx[i-1]=j
                break
        else:
            unique_snapshot_idx.append(i)
            match_idx[i-1]=i

    tmp_stable_configs=[]

    for i in range(sampling_last,n_images-1):
        if is_stable_enough(match_idx, i, sampling_last, criteria=criteria):
            tmp_stable_configs.append(i+1)

    stable_configs=[]
    stable_configs_matching_idx=[]
    for i in reversed(tmp_stable_configs):
        if match_idx[i-1] in stable_configs_matching_idx:
            continue

        stable_configs_matching_idx.append(match_idx[i-1])
        stable_configs.append(i)

    return stable_configs


def get_stable_config_tmp(Image):
    '''
    Just extract the 10% and 60% of the total number of images
    '''
    n_images = len(Image.dump)
    result = [int(np.round(n_images * 0.1)), int(np.round(n_images * 0.6))]
    return result


def main():
    path_dir = sys.argv[1]

    with open("to_rlx.dat",'w') as rlx_fo: #initializing the data file
        rlx_fo.write("#incidence,snapshots_idx\n")

    # nions = 50  # number of ion incidences
    nions = 20  # number of ion incidences
    # sampling_last = 100  # check length for stable configuration
    # criteria = 0.9  # check criteria for stable configuration

    for i in range(1, nions+1):
        print(i)
        Image=atom_image()
        Image.set_image_path(f"{path_dir}/dump_{i}.lammps")
        Image.read_dump()
        Image.get_graph()

        with open(f"{path_dir}/graph_{i}.pickle", 'wb') as O:
            pickle.dump(Image,O)

        with open("to_rlx.dat",'a') as rlx_fo:
            # stable_configs = get_stable_config(Image, criteria=criteria,
            #                                    sampling_last=sampling_last)
            stable_configs = get_stable_config_tmp(Image)
            line = f"{i} " + " ".join(map(str, stable_configs))
            rlx_fo.write(line)
            rlx_fo.write("\n")


if __name__ == '__main__':
    main()
