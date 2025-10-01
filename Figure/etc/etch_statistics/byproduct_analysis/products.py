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

            'NCF': 'FCN',
            'NCH': 'HCN',
            'H2': 'H$_2$',
            'N2': 'N$_2$',
            'NH3': 'NH$_3$',
            'CH4': 'CH$_4$',
            'SiHF3': 'SiHF$_3$',
            'SiH2F2': 'SiH$_2$F$_2$',
            }


class AbstractGeneratedProductsPlotter(ABC):
    def __init__(self, mol_dict, name, elem_dict):
        self.mol_dict = mol_dict
        self.name = name
        self.elem_dict = elem_dict

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
        n_incidence = max([i for i in mol_dict.values() for i in i]) + 1
        mat = np.zeros((len(mol_dict), n_incidence))
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

    def _stoichiometry_to_str(self, stoichiometry):
        elem_dict = self.elem_dict
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
        plt.rcParams.update({'font.size': 10, 'font.family': 'arial',})
        fig, ax_stackplot = plt.subplots(figsize=(3.5, 3.5))

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
        legend = ax_stackplot.legend(loc='upper center',
                                     bbox_to_anchor=(0.5, -0.4),
                                     # ncol=len(labels)/2,
                                     ncol=int(np.sqrt(len(labels))),
                                     fontsize='small',
                                     frameon=False)

        for patch in legend.get_patches():
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)

        # Give enough space for the legend
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.4)
        fig.savefig(self._get_output_filename(), bbox_inches='tight', dpi=200)

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
    if len(sys.argv) != 4:
        print("Usage: python products.py <mol_dict> <name> <SiO2/Si3N4>")
        sys.exit(1)
    mol_dict = sys.argv[1]
    name = sys.argv[2]
    system = sys.argv[3]

    if system == 'SiO2':
        elem_dict = {0: 'Si', 1: 'O', 2: 'C', 3: 'H', 4: 'F'}
    elif system == 'Si3N4':
        elem_dict = {0: 'Si', 1: 'N', 2: 'C', 3: 'H', 4: 'F'}
    else:
        print("Invalid system. Choose 'SiO2' or 'Si3N4'.")
        sys.exit(1)

    plotter_external = GeneratedProductsPlotterExternal(mol_dict, name, elem_dict)
    plotter_external.run()

if __name__ == "__main__":
    main()
