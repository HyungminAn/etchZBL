import sys
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import cycle
from itertools import islice
import pickle


@dataclass
class PARAMS:
    GAS_CONVERT_DICT = {
            'SiF4': 'SiF$_4$',
            'OC': 'CO',
            'O2C': 'CO$_2$',
            'OCF2': 'COF$_2$',
            'CF2': 'CF$_2$',
            'O2': 'O$_2$',
            'SiF2': 'SiF$_2$',
            'CF4': 'CF$_4$',
            'OF2': 'OF$_2$',
            'F2': 'F$_2$',
            }


class AbstractGeneratedProductsPlotter(ABC):
    def __init__(self, mol_dict, n_incidence, name):
        self.mol_dict = mol_dict
        self.n_incidence = n_incidence
        self.elem_dict = {0: 'Si', 1: 'O', 2: 'C', 3: 'H', 4: 'F'}
        self.name = name

    def run(self):
        mol_dict = self._get_molecule_dict()
        stack_mat = self._get_stack_matrix(mol_dict)
        labels = self._get_labels(mol_dict)
        self._plot(stack_mat, labels)
        self._save_data(mol_dict)

    @abstractmethod
    def _get_molecule_dict(self):
        pass

    def _get_stack_matrix(self, mol_dict):
        mat = np.zeros((len(mol_dict), self.n_incidence + 1))
        order = {
            k: idx
            for idx, (k, v) in enumerate(
                sorted(mol_dict.items(), key=lambda item: len(item[1]), reverse=True))
        }
        for stoichiometry, incidence_list in mol_dict.items():
            idx = order[stoichiometry]
            for incidence in incidence_list:
                mat[idx, incidence:] += 1
        return mat

    # def _get_molecule_dict_from_delete_log(self):
    #     mol_dict = defaultdict(list)
    #     file_path = f"{self.src}/delete.log"
    #     with open(file_path, "r") as f:
    #         for line in f:
    #             current_incidence = int(line.split()[1])
    #             stoichiometry = tuple(map(int, line.replace(',', '').split()[4:9]))
    #             mol_dict[stoichiometry].append(current_incidence)
    #     return mol_dict

    @staticmethod
    def _stoichiometry_to_str(stoichiometry):
        elem_dict = {0: 'Si', 1: 'O', 2: 'C', 3: 'H', 4: 'F'}
        return ''.join(f"{elem_dict[i]}{count}"
                       if count > 1 else f"{elem_dict[i]}"
                       for i, count in enumerate(stoichiometry)
                       if count > 0)

    def _get_labels(self, mol_dict):
        order = {
            k: idx
            for idx, (k, v) in enumerate(
                sorted(mol_dict.items(), key=lambda item: len(item[1]), reverse=True))
        }
        labels = [self._stoichiometry_to_str(k)
                  for (k, _) in sorted(order.items(), key=lambda x: x[1])]
        trunc_len = 10
        if len(labels) > trunc_len:
            labels = labels[:trunc_len] + [f'_{i}' for i in labels[trunc_len:]]
        return labels

    def _plot(self, stack_mat, labels, set_alpha=False):
        plt.rcParams.update({'font.size': 14})
        fig, ax_stackplot = plt.subplots(figsize=(6, 6))

        x = np.arange(stack_mat.shape[1])
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = islice(cycle(color_list), len(labels))
        hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'] * 100
        labels = [PARAMS.GAS_CONVERT_DICT.get(label, label) for label in labels]
        stack_collection = ax_stackplot.stackplot(x, stack_mat, labels=labels, colors=colors)

        for stack, label, hatch in zip(stack_collection, labels, hatches[:len(labels)]):
            stack.set_hatch(hatch)
            alpha = 1
            if set_alpha and 'Si' not in label:
                alpha = 0.1
            stack.set_alpha(alpha)
            stack.set_edgecolor((0, 0, 0, alpha))

        ax_stackplot.set_xlabel("Number of incidence")
        ax_stackplot.set_title(self._get_plot_title())
        legend = ax_stackplot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4),
                                     ncol=len(labels)/2, fontsize='small')

        for patch in legend.get_patches():
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)

        # Give enough space for the legend
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        fig.savefig(self._get_output_filename(), bbox_inches='tight')

    @abstractmethod
    def _get_plot_title(self):
        pass

    @abstractmethod
    def _get_output_filename(self):
        pass

    def _save_data(self, mol_dict):
        path_save = self._get_path_save_data()
        mol_list = []
        for stoichiometry, incidence_list in mol_dict.items():
            key = self._stoichiometry_to_str(stoichiometry)
            n_atoms = sum(stoichiometry)
            value = len(incidence_list)
            mol_list.append((key, n_atoms, value))
        mol_list = sorted(mol_list, key=lambda x: x[2], reverse=True)
        with open(path_save, "w") as f:
            f.write("Molecule, N_atoms, N_incidence\n")
            for key, n_atoms, value in mol_list:
                f.write(f"{key}, {n_atoms}, {value}\n")

    @abstractmethod
    def _get_path_save_data(self):
        pass


# class GeneratedProductsPlotterOriginal(AbstractGeneratedProductsPlotter):
#     def _get_molecule_dict(self):
#         return self._get_molecule_dict_from_delete_log()

#     def _get_plot_title(self):
#         return "Number of product molecules (Original)"

#     def _get_output_filename(self):
#         return 'products_analysis_original.png'

#     def _get_path_save_data(self):
#         return f"products_analysis_original_save.data"


# class GeneratedProductsPlotterFromDesorption(AbstractGeneratedProductsPlotter):
#     def _get_molecule_dict(self):
#         mol_dict_1 = self._get_molecule_dict_from_delete_log()
#         mol_dict_2 = self._get_molecule_dict_from_desorption_graph()
#         keys = set(mol_dict_1.keys()) | set(mol_dict_2.keys())
#         mol_dict = {key: mol_dict_1.get(key, []) + mol_dict_2.get(key, [])
#                     for key in keys}
#         return mol_dict

#     def _get_molecule_dict_from_desorption_graph(self):
#         mol_dict = defaultdict(list)
#         file_path = f"desorption_graph.dat"
#         with open(file_path, "r") as f:
#             lines = f.readlines()

#         for line in lines:
#             if line.startswith("--"):
#                 continue
#             parts = line.split('/')
#             incidence = int(parts[0])
#             composition = tuple(map(int, parts[1].split()))

#             mol_dict[composition].append(incidence)
#         return mol_dict

#     def _get_plot_title(self):
#         return "Number of product molecules (From Desorption Graph)"

#     def _get_output_filename(self):
#         return 'products_analysis_from_desorption.png'

#     def _get_path_save_data(self):
#         return f"products_analysis_from_desorption_save.data"


class GeneratedProductsPlotterExternal(AbstractGeneratedProductsPlotter):
    def _get_molecule_dict(self):
        with open(self.mol_dict, 'rb') as f:
            mol_dict = pickle.load(f)
        return mol_dict

    def _get_plot_title(self):
        return f"# products ({self.name})"

    def _get_output_filename(self):
        return f'result_{self.name}.png'

    def _get_path_save_data(self):
        return f"result_{self.name}.data"


def main():
    mol_dict = sys.argv[1]
    n_incidence = int(sys.argv[2])
    name = sys.argv[3]

    # plotter_original = GeneratedProductsPlotterOriginal(src, n_incidence)
    # plotter_desorption = GeneratedProductsPlotterFromDesorption(src, n_incidence)
    plotter_external = GeneratedProductsPlotterExternal(mol_dict, n_incidence, name)

    # plotter_original.run()
    # plotter_desorption.run()
    plotter_external.run()


if __name__ == "__main__":
    main()
