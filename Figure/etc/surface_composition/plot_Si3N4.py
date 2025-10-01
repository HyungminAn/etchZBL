import os
import pickle
from functools import wraps
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.io import read

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@dataclass
class PARAMS:
    elements = ['Si', 'N', 'C', 'H', 'F']
    elements_selected = ['Si', 'C', 'F']
    color_dict = {
        'Si': '#F0C8A0',
        'O': '#FF0D0D',
        'C': '#909090',
        'H': '#FFFFFF',
        'F': '#90E050',
    }

    LAMMPS_READ_OPTS = {
        'format': "lammps-data",
        'Z_of_type': {
            1: 14,
            2: 7,
            3: 6,
            4: 1,
            5: 9
        }
    }

    ion_convert_dict = {
        'CF': 'CF$^{+}$',
        'CH2F': 'CH$_{2}$F$^{+}$',
    }

class DataLoader:
    def run(self):
        data_dict = {
            'CF': {
                'path_exp': "exp/CF",
                'src': "/data_etch/gasplant63/chf_etch/chf_etch/CF_1000eV",
                },
            'CH2F': {
                'path_exp': "exp/CH2F",
                'src': "/data_etch/gasplant63/chf_etch/chf_etch/CH2F_1000eV",
                },
            }
        step = 100
        thickness = 40  # angstrom
        result = {}
        for label, data in data_dict.items():
            path_exp = data['path_exp']
            src = data['src']
            if os.path.exists(src):
                files = [os.path.join(src, f) for f in os.listdir(src)
                         if f.startswith("str_shoot_") and f.endswith("_after_mod.coo")]
                files = sorted(files, key=lambda x: int(x.split("_")[-3]))
            else:
                files = []
            data = self.get_data(files, step, thickness, f"{label}_data.pkl")
            data = self.post_process(data)
            data_exp = self.get_data_exp(path_exp)
            result[label] = {
                'data': data,
                'data_exp': data_exp,
            }
        return result

    @timeit
    def read_coo(self, file):
        return read(file, **PARAMS.LAMMPS_READ_OPTS)

    @timeit
    def filter_atoms(self, atoms, thickness):
        elements = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()[:, 2]
        max_val = np.max(pos)
        filtered = np.where((pos <= max_val) & (pos >= max_val - thickness))[0]
        elements = elements[filtered]
        return elements

    @timeit
    def count_elements(self, elements):
        count = {}
        for elem in elements:
            count[elem] = np.count_nonzero(elements == elem)
        return count

    def get_data(self, files, step, thickness, path_save):
        if os.path.exists(path_save):
            with open(path_save, "rb") as f:
                total_dict = pickle.load(f)
            return total_dict

        total_dict = {}
        for file in files[::step]:
            index = int(file.split("_")[-3])
            atoms = self.read_coo(file)
            elements = self.filter_atoms(atoms, thickness)
            count = self.count_elements(elements)
            total_dict[index] = count
            print("index: ", index)

        with open(path_save, "wb") as f:
            pickle.dump(total_dict, f)

        return total_dict

    @timeit
    def post_process(self, data):
        keys = sorted(data.keys())
        normalize_factor = data[0]['Si']
        total_dict = {}
        for key in keys:
            total_dict[key] = {elem: data[key][elem] / normalize_factor for elem in data[key].keys()}
        return total_dict

    @timeit
    def get_data_exp(self, path_exp):
        total_dict = {}
        for elem in PARAMS.elements:
            if not os.path.exists(os.path.join(path_exp, f"{elem}.csv")):
                print(f"Warning: {elem}.csv not found in {path_exp}")
                continue
            data = np.loadtxt(os.path.join(path_exp, f"{elem}.csv"), delimiter=",")
            total_dict[elem] = data
        return total_dict

class Plotter:
    def run(self, data):
        fig, axes = self.generate_figure(data)
        line_dict = {}

        for (ax, (label, data_dict)) in zip(axes, data.items()):
            data = data_dict['data']
            data_exp = data_dict['data_exp']

            keys = sorted(data.keys())

            for elem in PARAMS.elements_selected:
                x_exp, y_exp = data_exp[elem][:, 0], data_exp[elem][:, 1]
                x = np.array([k/9000 for k in keys])
                y = np.array([data[k].get(elem, 0) for k in keys])
                color = PARAMS.color_dict[elem]
                line, = ax.plot(x, y, label=elem, color=color)
                line_exp, = ax.plot(x_exp, y_exp, label=f"{elem} (exp)",
                                    color=color, marker='o', linewidth=0)
                self.decorate(ax, label)
                line_dict[elem] = line
                line_dict[f"{elem}_exp"] = line_exp

        self.set_global_legend(fig, axes, line_dict)
        self.add_figure_numbers(axes)
        self.save(fig)

    def generate_figure(self, data):
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            })
        n_data = len(data)
        fig, axes = plt.subplots(n_data, 1, figsize=(3.5, n_data * 2.5))
        # fig, axes = plt.subplots(n_data, 1, figsize=(4.5, n_data * 3))
        return fig, axes

    def add_figure_numbers(self, axes):
        ax_top, ax_bottom = axes[0], axes[-1]
        ax_top.text(-0.2, 1.2, "(a)", transform=ax_top.transAxes, fontsize=10)
        ax_bottom.text(-0.2, 1.2, "(b)", transform=ax_bottom.transAxes, fontsize=10)

    def decorate(self, ax, label):
        ax.set_xlabel("Ion dose (" + r"$\times$ " + "10$^{17}$ cm$^{-2}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(PARAMS.ion_convert_dict[label] + ', 1000 eV on Si$_3$N$_4$', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.0)

        ax_twin = ax.twinx()
        ax_twin.set_xlim(0, 1)
        ax_twin.set_ylim(0, 2.0)
        ax_twin.set_ylabel("Relative amount of atoms")

    def set_global_legend(self, fig, axes, line_dict):
        elem_list = PARAMS.elements_selected
        plotLines = []

        group_label = Line2D([], [], color='none', label='Ito et al.', linewidth=0)
        plotLines.append(group_label)
        for i in elem_list:
            plotLines.append(line_dict[f'{i}_exp'])
        group_label = Line2D([], [], color='none', label='This study', linewidth=0)
        plotLines.append(group_label)
        for i in elem_list:
            plotLines.append(line_dict[i])

        labels = [line.get_label() for line in plotLines]
        labels = [i.replace(' (exp)', '') for i in labels]
        # axes[1].legend(plotLines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.3), frameon=False)
        fig.legend(plotLines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False)

    def save(self, fig):
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        name = '3_1_4_valid_surface_composition_Si3N4'
        fig.savefig(f"{name}.png")
        fig.savefig(f"{name}.pdf")
        fig.savefig(f"{name}.eps")

@timeit
def main():
    dl = DataLoader()
    result = dl.run()
    pl = Plotter()
    pl.run(result)


if __name__ == "__main__":
    main()
