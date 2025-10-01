import os
import sys
import time
import pickle
import numpy as np
from ase.io import read
from ase.geometry import get_distances
from graph_tool import Graph
from graph_tool.topology import label_components

class AtomImage:
    def __init__(self):
        self.bond_length = self._generate_bondlength_matrix()
        self.elem_dict = {
            'Si': 0,
            'O': 1,
            'C': 2,
            'H': 3,
            'F': 4
        }
        self.n_elems = len(self.elem_dict)
        self.desorbed_ids_at_moment = {}
        self.deleting_moments = []
        self.NN_in_snapshot = {}
        self.image_path = None
        self.dump = None
        self.n_atoms_init = None
        self.image_idx = None
        self.initial_image = None

    def _generate_bondlength_matrix(self):
        bond_length = np.array([
            [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
            [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
            [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
            [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
            [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
        ])
        return bond_length + bond_length.T - np.diag(bond_length.diagonal())

    def set_images(self, image_path, i):
        self.image_path = image_path
        self.dump = read(f"{image_path}/dump_{i}.lammps", format='lammps-dump-text', index=":")
        self.initial_image = self.dump[0]
        self.n_atoms_init = len(self.initial_image)
        self.image_idx = i

    def find_deleted_atoms(self):
        n_image_dump = len(self.dump)
        results = [
            self._find_deleted_atoms_id(i)
            for i in range(1, n_image_dump)
        ]
        for (snapshot_idx, deleted_atom_ids) in results:
            if not deleted_atom_ids:
                continue
            self.deleting_moments.append(snapshot_idx)
            self.desorbed_ids_at_moment[snapshot_idx] = deleted_atom_ids

    def _find_deleted_atoms_id(self, snapshot):
        image_now = self.dump[snapshot]
        image_previous = self.dump[snapshot-1]
        id_diff = np.setdiff1d(image_previous.arrays['id'], image_now.arrays['id']).tolist()
        return (snapshot-1, id_diff)

    def find_NN_dump(self, dump_idx):
        start = time.time()
        dump = self.dump[dump_idx]
        n_atoms = len(dump)
        _, D_len = get_distances(dump.get_positions(), cell=dump.get_cell(), pbc=True)
        np.fill_diagonal(D_len, np.inf)
        symbols = dump.get_chemical_symbols()
        result = self._optimize_bond_length_matrix(symbols)
        compare = D_len < result
        self.NN_in_snapshot[dump_idx] = [(i, np.where(compare[i])[0]) for i in range(n_atoms)]
        end = time.time()
        print(f"dump_idx: {dump_idx} processing time: {end-start}")

    def _optimize_bond_length_matrix(self, symbols):
        idxs = np.array([self.elem_dict[symbol] for symbol in symbols])
        return self.bond_length[np.ix_(idxs, idxs)]

class ClusterAnalyzer:
    def __init__(self, atom_image):
        self.atom_image = atom_image
        self.cluster_by_id = []
        self.cluster_stoichiometry = []
        self.cluster_atom_types = []
        self.graph = None
        self.fully_desorbed_clusters = []
        self.partially_desorbed_clusters = []

    def find_desorbed_clusters(self, dump_idx, is_last_dump):
        image_dump = self.atom_image.dump[dump_idx]
        del_ids = self.atom_image.desorbed_ids_at_moment[dump_idx]
        # desorbed_idx_at_moment = np.array([np.argwhere(image_dump.arrays['id'] == i)[0][0] for i in del_ids])
        NN_list = self.atom_image.NN_in_snapshot[dump_idx]
        cluster, cluster_idx = self.get_cluster_from_graph(dump_idx, NN_list)

        for cluster_tmp in cluster_idx:
            atom_current_idx = np.argwhere(cluster.a == cluster_tmp).flatten()
            cluster_ids = image_dump.arrays['id'][atom_current_idx]
            desorbed_ids = set(cluster_ids) & set(del_ids)

            if len(desorbed_ids) == 0:
                continue

            if len(desorbed_ids) == len(cluster_ids):
                self.cluster_by_id.append(cluster_ids)
                stoichio_tmp = np.array([self.atom_image.elem_dict[i] for i in image_dump.symbols[atom_current_idx]])
                composition = np.bincount(stoichio_tmp, minlength=self.atom_image.n_elems)
                self.cluster_stoichiometry.append(composition)
                self.cluster_atom_types.append([image_dump.get_chemical_symbols()[i] for i in atom_current_idx])
                self.fully_desorbed_clusters.append({
                    'image_idx': self.atom_image.image_idx,
                    'dump_idx': dump_idx,
                    'cluster_ids': list(cluster_ids),
                    'composition': composition,
                    'atom_types': [image_dump.get_chemical_symbols()[i] for i in atom_current_idx]
                })
            elif len(desorbed_ids) > 0 or is_last_dump:
                remaining_ids = set(cluster_ids) - desorbed_ids
                stoichio_tmp = np.array([self.atom_image.elem_dict[i] for i in image_dump.symbols[atom_current_idx]])
                composition = np.bincount(stoichio_tmp, minlength=self.atom_image.n_elems)
                # composition = [image_dump.get_chemical_symbols()[i] for i in atom_current_idx]
                self.partially_desorbed_clusters.append({
                    'image_idx': self.atom_image.image_idx,
                    'dump_idx': dump_idx,
                    'cluster_idx': len(self.partially_desorbed_clusters) + 1,
                    'total_atoms': len(cluster_ids),
                    'desorbed': len(desorbed_ids),
                    'remaining': len(remaining_ids),
                    'desorbed_ids': list(desorbed_ids),
                    'remaining_ids': list(remaining_ids),
                    'composition': composition,
                    'atom_types': [image_dump.get_chemical_symbols()[i] for i in atom_current_idx]
                })

    def check_and_update_partially_desorbed(self):
        fully_desorbed_ids = set()

        for cluster in self.fully_desorbed_clusters:
            fully_desorbed_ids.update(cluster['cluster_ids'])

        for cluster in self.cluster_by_id:
            fully_desorbed_ids.update(cluster)

        for cluster in self.partially_desorbed_clusters:
            fully_desorbed_ids.update(cluster['desorbed_ids'])

        updated_partially_desorbed = []
        new_fully_desorbed = []
        for cluster in self.partially_desorbed_clusters:
            remaining_ids = set(cluster['remaining_ids'])
            if remaining_ids.issubset(fully_desorbed_ids):
                new_fully_desorbed.append({
                    'image_idx': cluster['image_idx'],
                    'dump_idx': cluster['dump_idx'],
                    'cluster_ids': cluster['desorbed_ids'] + cluster['remaining_ids'],
                    'composition': cluster['composition'],
                    'atom_types': cluster['atom_types'],
                })
            else:
                updated_partially_desorbed.append(cluster)

        self.partially_desorbed_clusters = updated_partially_desorbed
        self.update_fully_desorbed_clusters(new_fully_desorbed)

    def update_fully_desorbed_clusters(self, new_clusters):
        all_clusters = self.fully_desorbed_clusters + new_clusters
        updated_fully_desorbed = []

        for i, cluster in enumerate(all_clusters):
            cluster_id_set = set(cluster['cluster_ids'])
            is_subset = False

            for j, other_cluster in enumerate(all_clusters):
                if i != j:
                    other_id_set = set(other_cluster['cluster_ids'])
                    if cluster_id_set.issubset(other_id_set) and cluster_id_set != other_id_set:
                        is_subset = True
                        break

            if is_subset:
                continue

            updated_fully_desorbed.append(cluster)

        self.fully_desorbed_clusters = updated_fully_desorbed

    def get_cluster_from_graph(self, dump_idx, NN_list):
        '''
        Get cluster from graph
        '''
        graph = Graph(directed=False)
        n_atoms = len(self.atom_image.dump[dump_idx])
        graph.add_vertex(n_atoms)
        for i, neighbors in NN_list:
            for j in neighbors:
                graph.add_edge(i, j)
        cluster, hist = label_components(graph)
        slab_idx = np.argmax(hist)
        cluster_idx = [i for i in range(len(hist)) if i != slab_idx]

        self.graph = graph
        return cluster, cluster_idx

class DataProcessor:
    def __init__(self, path_image, n_incidence):
        self.path_image = path_image
        self.n_incidence = n_incidence

    def run(self):
        self.set_NN()
        self.check_cluster()

    def set_NN(self):
        os.makedirs("images", exist_ok=True)
        for i in range(1, self.n_incidence + 1):
            path_save = f"images/Images_{i}.bin"
            if os.path.exists(path_save):
                continue
            image = AtomImage()
            image.set_images(self.path_image, i)
            image.find_deleted_atoms()
            for moment_idx in image.deleting_moments:
                image.find_NN_dump(moment_idx)
            with open(path_save, 'wb') as f:
                pickle.dump(image, f)
            print(f'{path_save} written')

    def check_cluster(self):
        image_prev = None
        for i in range(1, self.n_incidence + 1):
            path_load = f"images/Images_{i}.bin"
            with open(path_load, 'rb') as f:
                image = pickle.load(f)
            is_atom_reset = self.check_atom_reset(i, image_prev, image)

            analyzer = ClusterAnalyzer(image)
            for j, moment_idx in enumerate(image.deleting_moments):
                is_last_dump = j == len(image.deleting_moments) - 1
                analyzer.find_desorbed_clusters(moment_idx, is_last_dump)
            analyzer.check_and_update_partially_desorbed()
            self.write_desorbed_cluster_info(i,
                                             analyzer.fully_desorbed_clusters,
                                             is_atom_reset)
            self.write_partially_desorbed_cluster_info(analyzer.partially_desorbed_clusters)
            image_prev = image

    @staticmethod
    def numpyarr2str(arr):
        return " ".join(np.char.mod('%d', arr))

    @staticmethod
    def find_new_desorbed_clusters(image, image_prev):
        if image_prev is None:
            return image.cluster_stoichiometry, image.cluster_by_id, image.cluster_atom_types
        n_clusters_prev = len(image_prev.cluster_stoichiometry)
        n_clusters_now = len(image.cluster_stoichiometry)
        if n_clusters_now > n_clusters_prev:
            return (image.cluster_stoichiometry[n_clusters_prev:],
                    image.cluster_by_id[n_clusters_prev:],
                    image.cluster_atom_types[n_clusters_prev:])
        elif n_clusters_now < n_clusters_prev:
            return image.cluster_stoichiometry, image.cluster_by_id, image.cluster_atom_types
        else:
            return [], [], []

    @staticmethod
    def write_desorbed_cluster_info(i, fully_desorbed_clusters, is_atom_reset):
        path_save = "desorption_graph.dat"
        # if os.path.exists(path_save):
        #     print(f"{path_save} exists. Skipping {i}...")
        #     return

        with open(path_save, 'a') as f:
            if is_atom_reset:
                f.write("-- atom id reset --\n")
            for cluster in fully_desorbed_clusters:
                comp = DataProcessor.numpyarr2str(cluster['composition'])
                cluster_ids = DataProcessor.numpyarr2str(cluster['cluster_ids'])
                atom_types = " ".join(cluster['atom_types'])
                f.write(f"{i} / {comp} / {cluster_ids} / {atom_types}\n")
        print(f"images/Images_{i}.bin Complete")

    @staticmethod
    def write_partially_desorbed_cluster_info(partially_desorbed_clusters):
        path_save = "partially_desorbed_clusters.dat"
        # if os.path.exists(path_save):
        #     return

        with open(path_save, 'a') as f:
            for cluster in partially_desorbed_clusters:
                f.write(f"Image {cluster['image_idx']}, Dump {cluster['dump_idx']}, Cluster {cluster['cluster_idx']}: "
                        f"Total atoms {cluster['total_atoms']}, Desorbed {cluster['desorbed']}, Remaining {cluster['remaining']}\n")
                f.write(f"Desorbed IDs: {' '.join(map(str, cluster['desorbed_ids']))}\n")
                f.write(f"Remaining IDs: {' '.join(map(str, cluster['remaining_ids']))}\n")
                comp = DataProcessor.numpyarr2str(cluster['composition'])
                f.write(f"Composition: {comp}\n\n")

    @staticmethod
    def check_atom_reset(i, image_prev, image):
        '''
        Check whether atom id is reset or not
        by comparing the number of atoms in the previous and current images
        (if slab is added, the number of atoms will increase by more than 10)
        '''
        if i > 1 and image_prev is not None:
            n_atoms_now = len(image.initial_image)
            n_atoms_prev = len(image_prev.initial_image)
            is_atom_id_reset = n_atoms_now - n_atoms_prev > 10
        else:
            is_atom_id_reset = True

        if is_atom_id_reset:
            image.cluster_by_id = []
            image.cluster_stoichiometry = []
            image.cluster_atom_types = []
            print('reset')
        else:
            image.cluster_by_id = image_prev.cluster_by_id.copy()
            image.cluster_stoichiometry = image_prev.cluster_stoichiometry.copy()
            image.cluster_atom_types = image_prev.cluster_atom_types.copy()
        return is_atom_id_reset

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_image> <n_incidence>")
        sys.exit(1)

    path_image = sys.argv[1]
    n_incidence = int(sys.argv[2])

    runner = DataProcessor(path_image, n_incidence)
    runner.run()
