from dataclasses import dataclass

import numpy as np

from ase.neighborlist import neighbor_list
import graph_tool.all as gt
from graph_tool.topology import label_components


@dataclass
class PARAMS:
    @dataclass
    class SYSTEM_DEPENDENT:
        bond_length = np.array([
            [2.62476, 1.75847, 1.98436, 1.77338, 1.88066],
            [1.75847, 1.41798, 1.31503, 1.1351 , 1.57798],
            [1.98436, 1.31503, 1.51158, 1.30897, 1.49227],
            [1.77338, 1.1351 , 1.30897, 0.86282, 1.07879],
            [1.88066, 1.57798, 1.49227, 1.07879, 1.63796],
        ])
        atom_idx = {
            0: 'Si',
            1: 'O',
            2: 'C',
            3: 'H',
            4: 'F',
        }
        ELEM_LIST = ['Si', 'O', 'C', 'H', 'F']
        LAMMPS_SAVE_OPTS = {
            'format': 'lammps-data',
            'atom_style': 'atomic',
            'specorder': ELEM_LIST,
        }
        gas_dict = {
            "HF": np.array([0,0,0,1,1], dtype=int),
            "O2": np.array([0,2,0,0,0], dtype=int),
            "SiF2": np.array([1,0,0,0,2], dtype=int),
            "SiF4": np.array([1,0,0,0,4], dtype=int),
            "CO": np.array([0,1,1,0,0], dtype=int),
            "CF": np.array([0,0,1,0,1], dtype=int),
            "CO2": np.array([0,2,1,0,0], dtype=int),
            "SiOF2": np.array([1,1,0,0,2], dtype=int),

            "Si": np.array([1,0,0,0,0], dtype=int),
            "O": np.array([0,1,0,0,0], dtype=int),
            "C": np.array([0,0,1,0,0], dtype=int),
            "H": np.array([0,0,0,1,0], dtype=int),
            "F": np.array([0,0,0,0,1], dtype=int),
        }


    LAMMPS_READ_OPTS = {
        "format": "lammps-data",
        "index": 0,
        "atom_style": "atomic"
    }
    VASP_SAVE_OPTS = {}

    @dataclass
    class DUMP_INFO:
        n_incidences = 20  # number of ion incidences
        ions = ['CF', 'CF3', 'CH2F']
        energies = [20, 50]

    path_incidences = '01_incidences'
    path_to_rlx = '01_to_rlx.dat'

    path_nnp_pickle = '02_nnp_rlx.pickle'
    path_nnp_extxyz = '02_nnp_rlx.extxyz'

    path_post_process = '03_data_to_relax'
    path_unique_bulk = '03_unique_bulk.dat'
    path_unique_bulk_extxyz = '03_unique_bulk.extxyz'
    path_unique_gas = '03_desorbed_gas_id.dat'

    path_reaction_data = '04_rxn.dat'

    gas_crit = 4.5
    delete_crit = 18
    fix_h = 2.0

def generate_bondlength_dict():
    bond_length_mat = PARAMS.SYSTEM_DEPENDENT.bond_length
    atom_idx = PARAMS.SYSTEM_DEPENDENT.atom_idx
    n_species, _ = bond_length_mat.shape
    bl_dict = {}
    for i in range(n_species):
        for j in range(i, n_species):
            key = (atom_idx[i], atom_idx[j])
            value = bond_length_mat[i, j]
            bl_dict[key] = value
    return bl_dict

class GraphAnalyzer:
    def __init__(self):
        self.bond_length_dict = generate_bondlength_dict()

    def set_image_path(self,image_path):
        self.image_path=image_path

    def run_batch(self, images):
        graphs, vprops, clusters, histograms = [], [], [], []
        for image in images:
            graph, vprop, cluster, hist = self.run_single(image)
            graphs.append(graph)
            vprops.append(vprop)
            clusters.append(cluster)
            histograms.append(hist)
        return graphs, vprops, clusters, histograms

    def run_single(self, image):
        graph = gt.Graph(directed=False)
        NN_list = neighbor_list('ij', image, self.bond_length_dict)
        vprop = graph.new_vertex_property("short", vals=image.get_atomic_numbers())
        graph.add_edge_list(zip(NN_list[0], NN_list[1]))
        cluster, hist = label_components(graph, directed=False)
        return graph, vprop, cluster, hist

    def is_two_graph_same(self, graph1, graph2):
        return gt.similarity(graph1, graph2, norm=False, distance=True, asymmetric=True) == 0.0

