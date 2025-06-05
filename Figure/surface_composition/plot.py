import os
import pickle
from functools import wraps
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
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
    elements = ['Si', 'O', 'C', 'F']
    color_dict = {
        'Si': 'red',
        'O': 'blue',
        'C': 'green',
        'F': 'orange',
    }

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

class DataLoader:
    def run(self):
        data_dict = {
            'CF3_50eV': {
                'path_exp': "exp/50eV",
                'src': "/data_etch/data_HM/nurion/set_2/CF3_50_coo/CF3/50eV",
                },
            'CF3_400eV': {
                'path_exp': "exp/400eV",
                'src': "/data2/andynn/ZBL_modify/250331_ContinueRun/NewRun/CF3_400/",
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
        total_dict = {}
        for key in keys:
            total = sum(data[key].values())
            total_dict[key] = {elem: data[key][elem] / total for elem in data[key].keys()}
        return total_dict

    @timeit
    def get_data_exp(self, path_exp):
        total_dict = {}
        for elem in PARAMS.elements:
            data = np.loadtxt(os.path.join(path_exp, f"{elem}.csv"), delimiter=",")
            data[:, 0] = data[:, 0] / 10   # convert to 10^17
            total_dict[elem] = data
        return total_dict

class Plotter:
    def run(self, data):
        fig, axes = self.generate_figure(data)

        for (ax, (label, data_dict)) in zip(axes, data.items()):
            data = data_dict['data']
            data_exp = data_dict['data_exp']

            keys = sorted(data.keys())
            for elem in PARAMS.elements:
                x_exp, y_exp = data_exp[elem][:, 0], data_exp[elem][:, 1]
                x = np.array([k/9000 for k in keys])
                y = np.array([data[k][elem] for k in keys])
                color = PARAMS.color_dict[elem]
                ax.plot(x, y, label=elem, color=color)
                ax.plot(x_exp, y_exp, label=f"{elem} (exp)", color=color, marker='o', linestyle='--')
                self.decorate(ax, label)

        # self.set_legends(axes)
        self.set_global_legend(fig, axes)
        self.save(fig)

    def generate_figure(self, data):
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            })
        n_data = len(data)
        fig, axes = plt.subplots(n_data, 1, figsize=(3.5, n_data * 3.5))
        # fig, axes = plt.subplots(n_data, 1, figsize=(4.5, n_data * 3))
        return fig, axes

    def decorate(self, ax, label):
        ax.set_xlabel("Ion dose (" + r"$\times$ " + "10$^{17}$cm$^{-2}$)")
        ax.set_ylabel("Atomic composition")
        ax.set_title(label.replace("CF3_", "CF${}_{3}^{+}$, ") + ' on SiO$_2$',
                     fontsize=10)
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0, 0.8)
        # ax.legend(loc='upper right', ncol=2)

    def set_global_legend(self, fig, axes):
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, 0), frameon=False)

    # def set_legends(self, axes):
    #     for ax in axes:
    #         ax.legend(loc='upper right', ncol=4, frameon=False)

    def save(self, fig):
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        name = '3_1_4_valid_surface_composition_SiO2'
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
