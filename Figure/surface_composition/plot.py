import os
import sys
import time
import pickle
from functools import wraps
from dataclasses import dataclass

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.io import read

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@dataclass
class PARAMS:
    @dataclass
    class SiO2:
        elements = ['Si', 'O', 'C', 'F']
        LAMMPS_READ_OPTS = {
            'format': "lammps-data",
            'Z_of_type': {
                1: 14,
                2: 8,
                3: 6,
                4: 1,
                5: 9
            }
        }
        label = 'Toyoda et al.'
        # xlim = (0, 1.5)
        xlim = (0, 0.2)
        ylim = (0, 0.8)
        ylabel = 'Atomic composition'

        norm_by_init = False

    @dataclass
    class Si3N4:
        elements = ['Si', 'C', 'F']
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
        label = 'Ito et al.'
        xlim = (0, 1)
        ylim = (0, 2)
        ylabel = 'Relative amount of atoms'

        norm_by_init = True

    color_dict = {
        'Si': '#F0C8A0',
        'O': '#FF0D0D',
        'C': '#909090',
        'H': '#FFFFFF',
        'F': '#90E050',
    }

    # color_dict_test = {
    #     'Si': '#a88c70',
    #     'O': '#b20909',
    #     'C': '#646464',
    #     'F': '#649c38',
    # }
    color_dict_test = {
        'Si': '#f4d8bc',
        'O': '#ff5555',
        'C': '#929292',
        'F': '#b1e984',
    }

    elem_symbols = {
        'Si': 'o',
        'O': 's',
        'C': 'D',
        'F': 'v',
    }

    formula_convert_dict = {
        'CF': 'CF$^{+}$',
        'CF3': 'CF${}_{3}^{+}$',
        'CH2F': 'CH${}_{2}$F$^{+}$',

        'SiO2': 'SiO$_{2}$',
        'Si3N4': 'Si$_{3}$N$_{4}$',
    }

    step = 100
    thickness = 50  # angstrom
    normalize_factor = 9000
    fig_prefix = "3_1_4_valid_surface_composition"

class DataLoader:
    def __init__(self, params):
        self.params = params

    def run(self, path_yaml):
        with open(path_yaml, 'r') as f:
            data_dict = yaml.safe_load(f)
        step = PARAMS.step
        thickness = PARAMS.thickness
        result = {}
        for label, data in data_dict.items():
            path_exp = data['path_exp']
            files = self.get_file_list(data['src'])
            data = self.get_data(files, step, thickness, f"{label}_data.pkl")
            data = self.post_process(data)
            data_exp = self.get_data_exp(path_exp)
            result[label] = {
                'data': data,
                'data_exp': data_exp,
            }
        return result

    def get_file_list(self, src):
        result = []
        if isinstance(src, str):
            src = [src]
        for s in src:
            if not os.path.exists(s):
                continue
            files = [os.path.join(s, f) for f in os.listdir(s)
             if self.filter_filename(f)]
            result.extend(files)
        result = sorted(result, key=lambda x: int(x.split("_")[-3]))
        return result

    def filter_filename(self, filename):
        return filename.startswith("str_shoot_") and filename.endswith("_after_mod.coo")

    @timeit
    def read_coo(self, file):
        return read(file, **self.params.LAMMPS_READ_OPTS)

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
        total_dict = {}
        for key in keys:
            if self.params.norm_by_init:
                norm_factor = data[0]['Si']
            else:
                norm_factor = sum(data[key].values())
            total_dict[key] = {elem: data[key][elem] / norm_factor for elem in data[key].keys()}
        return total_dict

    @timeit
    def get_data_exp(self, path_exp):
        total_dict = {}
        if path_exp is None:
            print("No experimental data provided.")
            return total_dict
        for elem in self.params.elements:
            data = np.loadtxt(os.path.join(path_exp, f"{elem}.csv"), delimiter=",")
            data[:, 0] = data[:, 0] / 10   # convert to 10^17
            total_dict[elem] = data
        return total_dict

class Plotter:
    def __init__(self, system, params):
        self.system = system
        self.params = params

    def run(self, data):
        fig, axes = self.generate_figure(data)
        line_dict = {}

        for (ax, (label, data_dict)) in zip(axes, data.items()):
            self.plot_sim(ax, data_dict['data'], line_dict)
            if data_dict['data_exp']:
                self.plot_exp(ax, data_dict['data_exp'], line_dict)
            self.decorate(ax, label)

        self.set_global_legend(fig, line_dict)
        self.add_figure_numbers(axes)
        self.save(fig)

    def plot_sim(self, ax, data, line_dict):
        keys = sorted(data.keys())
        for elem in self.params.elements:
            x = np.array([k/PARAMS.normalize_factor for k in keys])
            y = np.array([data[k].get(elem, 0) for k in keys])
            color = PARAMS.color_dict[elem]
            line, = ax.plot(x, y, label=elem, color=color)
            line_dict[elem] = line

    def plot_exp(self, ax, data_exp, line_dict):
        for elem in self.params.elements:
            color = PARAMS.color_dict_test[elem]
            x_exp, y_exp = data_exp[elem][:, 0], data_exp[elem][:, 1]
            line_exp, = ax.plot(x_exp, y_exp, label=f"{elem} (exp)",
                                color=color, marker=PARAMS.elem_symbols[elem],
                                linewidth=0, markeredgecolor='black')
            line_dict[f"{elem}_exp"] = line_exp

    def generate_figure(self, data):
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            })
        n_data = len(data)
        fig, axes = plt.subplots(n_data, 1, figsize=(3.5, n_data * 3))
        # fig, axes = plt.subplots(n_data, 1, figsize=(4.5, n_data * 3))
        return fig, axes

    def add_figure_numbers(self, axes):
        ax_top, ax_bottom = axes[0], axes[-1]
        ax_top.text(-0.2, 1.2, "(a)", transform=ax_top.transAxes, fontsize=10)
        ax_bottom.text(-0.2, 1.2, "(b)", transform=ax_bottom.transAxes, fontsize=10)

    def decorate(self, ax, label):
        ax.set_xlabel("Ion dose (" + r"$\times$ " + "10$^{17}$ cm$^{-2}$)")
        ax.set_ylabel(self.params.ylabel)
        ax.set_title(self.get_title(label), fontsize=10)
        ax.set_xlim(self.params.xlim)
        ax.set_ylim(self.params.ylim)

    def get_title(self, label):
        ion, energy = label.split("_")
        ion = PARAMS.formula_convert_dict.get(ion, ion)
        system = PARAMS.formula_convert_dict.get(self.system, self.system)
        return f"{ion}, {energy} on {system}"

    def set_global_legend(self, fig, line_dict):
        plotLines = []
        elem_list = self.params.elements

        plot_exp = np.any([f'{i}_exp' in line_dict for i in elem_list])
        if plot_exp:
            group_label = Line2D([], [], color='none', label=self.params.label, linewidth=0)
            plotLines.append(group_label)
            for i in elem_list:
                if line_dict.get(f'{i}_exp'):
                    plotLines.append(line_dict[f'{i}_exp'])

        group_label = Line2D([], [], color='none', label='This study', linewidth=0)
        plotLines.append(group_label)
        for i in elem_list:
            plotLines.append(line_dict[i])

        labels = [line.get_label() for line in plotLines]
        labels = [i.replace(' (exp)', '') for i in labels]
        fig.legend(plotLines,
                   labels,
                   loc='lower center',
                   ncol=2 if plot_exp else 1,
                   bbox_to_anchor=(0.5, 0.0),
                   frameon=False)

    def save(self, fig):
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        name = f'{PARAMS.fig_prefix}_{self.system}'
        fig.savefig(f"{name}.png")
        fig.savefig(f"{name}.pdf")
        fig.savefig(f"{name}.eps")

@timeit
def main():
    if len(sys.argv) != 3:
        print("Usage: python plot.py <path_to_yaml> <SiO2/Si3N4>")
        sys.exit(1)

    path_yaml = sys.argv[1]
    system = sys.argv[2]

    if system == 'SiO2':
        params = PARAMS.SiO2
    elif system == 'Si3N4':
        params = PARAMS.Si3N4
    else:
        raise ValueError("System must be 'SiO2' or 'Si3N4'.")

    dl = DataLoader(params)
    result = dl.run(path_yaml)
    pl = Plotter(system, params)
    pl.run(result)

if __name__ == "__main__":
    main()
