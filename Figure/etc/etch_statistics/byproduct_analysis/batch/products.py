import sys
import pickle
from itertools import cycle
from itertools import islice
from dataclasses import dataclass

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
            'OCH2': 'CH$_2$O',
            'OCHF': 'CHFO',
            'OH2': 'H$_2$O',

            'NCF': 'FCN',
            'NCH': 'HCN',
            'H2': 'H$_2$',
            'N2': 'N$_2$',
            'NH3': 'NH$_3$',
            'CH4': 'CH$_4$',
            'SiHF3': 'SiHF$_3$',
            'SiH2F2': 'SiH$_2$F$_2$',
            }

    ION_CONVERT_DICT = {
            'CF': 'CF$^+$',
            'CF2': 'CF${}_{2}^{+}$',
            'CF3': 'CF${}_{3}^{+}$',
            'CH2F': 'CH${}_{2}$F$^+$',
            'CHF2': 'CHF${}_{2}^{+}$',
            }

class FigureGenerator:
    def run(self, paths):
        plt.rcParams.update({
            'font.size': 18,
            'font.family': 'arial',
            'hatch.linewidth': 0.5,
            })
        ion_energy_pair = {}
        for ion in paths.keys():
            ion_energy_pair[ion] = []
            for energy in paths[ion].keys():
                ion_energy_pair[ion].append(energy)
        n_row = len(ion_energy_pair)
        n_col = max(len(energies) for energies in ion_energy_pair.values())

        multiplier = 2.3
        fig_size = (7.1 * multiplier, 7.1 * multiplier)
        fig, axes = plt.subplots(n_row, n_col, figsize=fig_size,)
        ax_dict = {}
        for row_idx, ion in enumerate(ion_energy_pair.keys()):
            for energy in ion_energy_pair[ion]:
                ax = axes[row_idx, ion_energy_pair[ion].index(energy)]
                key = (ion, energy)
                ax_dict[key] = ax
        return fig, ax_dict

class StackMatrixBuilder:
    def run(self, mol_dict):
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
        return mat, order

class AxisProcessor:
    def run(self, stack_mat, labels, ax, decorate_x=False, decorate_y=False):
        x = np.arange(stack_mat.shape[1], dtype=float)
        x /= 9000
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = islice(cycle(color_list), len(labels))
        labels = [PARAMS.GAS_CONVERT_DICT.get(label, label) for label in labels]
        stack_collection = ax.stackplot(x, stack_mat, labels=labels, colors=colors)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10000)

        if decorate_x:
            ax.set_xlabel(r"Ion dose ($\times$ 10$^{17}$ cm$^{-2}$)")
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if decorate_y:
            ax.set_ylabel("Count")
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        return stack_collection

class FigureDecorator:
    def __init__(self, fig, ax_dict, stack_results, all_labels):
        """
        fig            : matplotlib.figure.Figure
        ax_dict        : {(ion,energy): Axes}
        stack_results  : {(ion,energy): [PolyCollection, ...]}
        all_labels     : {(ion,energy): [str, ...]}  # the global list of labels in plotting order
        """
        self.fig = fig
        self.ax_dict = ax_dict
        self.stack_results = stack_results
        self.all_labels = all_labels

        labels_set = set()
        for lb in self.all_labels.values():
            labels_set.update(lb)
        self.labels_set = sorted(labels_set)
        self.make_color_hatch_maps()

    def make_color_hatch_maps(self):
        # build a color map from the default cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        hatches = ['/', '*', '\\', '|', '-', '.', 'x', 'o', 'O', '+']
        i = 0

        self.color_map = {}
        self.hatch_map = {}
        for hatch in hatches:
            for color in colors:
                label = self.labels_set[i]
                self.color_map[label] = color
                self.hatch_map[label] = hatch
                print(i, label, color, hatch)
                i += 1
                if i >= len(self.labels_set):
                    return

    def run(self):
        # 1) style every stack-collection on every axis
        for key, stack_collection in self.stack_results.items():
            ax = self.ax_dict[key]
            labels = self.all_labels[key]
            for poly, label in zip(stack_collection, labels):
                poly.set_facecolor(  self.color_map[label] )
                poly.set_hatch(      self.hatch_map[label] )
                poly.set_edgecolor(  'black' )
                poly.set_alpha(      1.0 )

            # you can still set per‐axis labels/titles here if you want:
            ion, energy = key
            if ion in PARAMS.ION_CONVERT_DICT:
                ion = PARAMS.ION_CONVERT_DICT[ion]
            ax.set_title(f"{ion} – {energy} eV", fontsize=18)

        # 2) make one global legend on the figure
        handles = [
            Patch(facecolor=self.color_map[l],
                  hatch=self.hatch_map[l],
                  edgecolor='black',
                  label=PARAMS.GAS_CONVERT_DICT.get(l, l))
            for l in self.labels_set
        ]
        ncol = int(np.ceil(np.sqrt(len(self.labels_set))))
        self.fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=False,
            fontsize=18,
        )
        # 3) adjust layout so legend isn’t cut off
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.2)

class DataSaver:
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


class LabelConverter:
    def __init__(self, elem_dict):
        self.elem_dict = elem_dict

    def run(self, mol_dict):
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

    def _stoichiometry_to_str(self, stoichiometry):
        elem_dict = self.elem_dict
        return ''.join(f"{elem_dict[i]}{count}"
                       if count > 1 else f"{elem_dict[i]}"
                       for i, count in enumerate(stoichiometry)
                       if count > 0)

class Plotter:
    def __init__(self, elem_dict):
        self.elem_dict = elem_dict

    def run(self, paths):
        fg = FigureGenerator()
        fig, ax_dict = fg.run(paths)

        ion_energy_pair = {}
        for ion in paths.keys():
            ion_energy_pair[ion] = []
            for energy in paths[ion].keys():
                ion_energy_pair[ion].append(energy)

        smb = StackMatrixBuilder()
        lbc = LabelConverter(self.elem_dict)
        axp = AxisProcessor()
        results = {}
        labels_all = {}
        for ion in paths.keys():
            for energy, path in paths[ion].items():
                with open(path, 'rb') as f:
                    mol_dict = pickle.load(f)
                stack_mat, order = smb.run(mol_dict)
                labels = lbc.run(mol_dict)
                ax = ax_dict[(ion, energy)]

                decorate_x = (ion == [i for i in paths.keys()][-1])
                decorate_y = (energy == [i for i in ion_energy_pair[ion]][0])

                result = axp.run(stack_mat,
                                 labels,
                                 ax,
                                 decorate_x=decorate_x,
                                 decorate_y=decorate_y,
                                 )
                results[(ion, energy)] = result
                labels_all[(ion, energy)] = labels
                print(f"Processed {ion} at {energy} eV: {labels}")

        fd = FigureDecorator(fig, ax_dict, results, labels_all)
        fd.run()

        # Give enough space for the legend
        fig.savefig('byproducts.png', dpi=200)

        # self._save_data(mol_dict)

def main():
    if len(sys.argv) != 3:
        print("Usage: python products.py <path_yaml> <SiO2/Si3N4>")
        sys.exit(1)
    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        paths = yaml.safe_load(f)
    system = sys.argv[2]

    if system == 'SiO2':
        elem_dict = {0: 'Si', 1: 'O', 2: 'C', 3: 'H', 4: 'F'}
    elif system == 'Si3N4':
        elem_dict = {0: 'Si', 1: 'N', 2: 'C', 3: 'H', 4: 'F'}
    else:
        print("Invalid system. Choose 'SiO2' or 'Si3N4'.")
        sys.exit(1)

    pl = Plotter(elem_dict)
    pl.run(paths)

if __name__ == "__main__":
    main()
