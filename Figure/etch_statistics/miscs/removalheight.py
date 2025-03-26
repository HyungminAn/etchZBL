import sys
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from multiprocessing import Pool, cpu_count
from graph_tool import Graph
from graph_tool.topology import label_components

class RemovalHeightPlotter:
    def __init__(self, target_folder):
        self.target_folder = target_folder
        self.bond_length = self._generate_bondlength_matrix()
        self.elem_dict = {
                0: 'Si',
                1: 'O',
                2: 'C',
                3: 'H',
                4: 'F'
                }
        self.removal_heights = {}

    def run(self):
        self.analyze_removal_heights()
        self.plot_histograms()

    def _generate_bondlength_matrix(self):
        bond_length = np.array([
            [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
            [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
            [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
            [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
            [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
        ])
        return bond_length + bond_length.T - np.diag(bond_length.diagonal())

    def _find_nearest_neighbors(self, atoms, i):
        n_atoms = len(atoms)
        elem_idx = atoms.get_array('type')
        distances = atoms.get_distances(i, range(n_atoms), mic=True)
        neighbors = np.where(distances < self.bond_length[elem_idx[i]-1, elem_idx-1])
        return i, neighbors[0][neighbors[0] != i]

    def _get_slab_max_z(self, atoms):
        graph = Graph(directed=False)
        graph.add_vertex(len(atoms))

        with Pool(cpu_count()) as pool:
            nearest_neighbors = pool.starmap(self._find_nearest_neighbors, [(atoms, i) for i in range(len(atoms))])

        for idx, neighbors in nearest_neighbors:
            for j in neighbors:
                graph.add_edge(idx, j)

        cluster, hist = label_components(graph)
        slab_idx = np.argmax(hist)
        atom_in_slab = np.where(cluster.a == slab_idx)[0]
        return np.max(atoms.get_positions()[atom_in_slab, 2])

    def analyze_removal_heights(self):
        read_options = {
                'format': 'lammps-data',
                'style': 'atomic',
                'sort_by_id': False
                }
        with open(f"{self.target_folder}/delete.log", "r") as f:
            for line in f:
                parts = line.replace(',','').split()
                curr_inc = int(parts[1])
                del_idx_list = [int(i) for i in parts[11:]]
                stoichiometry = tuple(map(int, parts[4:9]))

                name = f"{self.target_folder}/CHF_shoot_{curr_inc}.coo"
                atoms = read(name, **read_options)
                max_z_slab = self._get_slab_max_z(atoms)
                max_z_molecule = np.max(atoms.get_positions()[del_idx_list, 2])
                buried_z = max_z_molecule - max_z_slab

                if stoichiometry not in self.removal_heights:
                    self.removal_heights[stoichiometry] = []
                self.removal_heights[stoichiometry].append(buried_z)

    def _stoichiometry_to_str(self, stoichiometry):
        return ''.join(f"{self.elem_dict[i]}{count}" if count > 1 else f"{self.elem_dict[i]}"
                       for i, count in enumerate(stoichiometry) if count > 0)

    def plot_histograms(self):
        plt.rcParams.update({'font.size': 18})
        n_types = len(self.removal_heights)
        fig_rows = (n_types + 1) // 2
        fig, axes = plt.subplots(fig_rows, 2, figsize=(20, 5*fig_rows))
        axes = axes.flatten()

        all_heights = np.concatenate(list(self.removal_heights.values()))
        z_min, z_max = np.floor(min(all_heights)), np.ceil(max(all_heights))
        n_bins = int(z_max - z_min)

        for ax, (stoichiometry, heights) in zip(axes, self.removal_heights.items()):
            label = self._stoichiometry_to_str(stoichiometry)
            ax.hist(heights, bins=n_bins, range=(z_min, z_max))
            ax.set_xlabel('Removal height (Å)')
            ax.set_ylabel('Count')
            ax.set_ylim(None, 30)
            ax.set_title(label)
            ax.axvline(0, color='grey', linestyle='--')

        # Plot total histogram
        ax = axes[-1]
        ax.hist(all_heights, bins=n_bins, range=(z_min, z_max))
        ax.set_xlabel('Removal height (Å)')
        ax.set_ylabel('Count')
        ax.set_title('All molecules')
        ax.axvline(0, color='grey', linestyle='--')

        fig.tight_layout()
        fig.savefig('removal_height_histograms.png')


if __name__ == '__main__':
    target_folder = sys.argv[1]
    plotter = RemovalHeightPlotter(target_folder)
    plotter.run()
