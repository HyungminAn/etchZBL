import os
import sys
import pickle
from functools import wraps
from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import yaml


class EnergyNotCalculatedError(Exception):
    pass


class pklSaver:
    @staticmethod
    def run(func_gen_name):
        '''
        Decorator to save the result of a function as a numpy file.
        '''
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                path_save = func_gen_name(self)
                if os.path.exists(path_save):
                    print(f"{path_save} already exists, loading data from it.")
                    with open(path_save, 'rb') as f:
                        data = pickle.load(f)
                    return data

                data = func(self, *args, **kwargs)
                with open(path_save, 'wb') as f:
                    pickle.dump(data, f)
                    print(f"Data saved to {path_save}")
                return data
            return wrapper
        return decorator

class DataLoader:
    def __init__(self, src, cal_type_1, cal_type_2):
        self.src = src
        self.cal_type_1 = cal_type_1
        self.cal_type_2 = cal_type_2

    def run(self, ion_type, ion_energy):
        '''
        example of `rxn.dat`:
            #reactants,/products,/add_to_rxn_E
            1_25/1_151,CO
            1_151,CF/2_25
            ...

        example of `rxn_list`:
            [
                ['1_25', '1_151,CO'],
                ['1_151,CF', '2_25'],
                ...
            ]
        '''
        src = f"{self.src}/{ion_type}/{ion_energy}"
        path_rxn_E = f"{src}/rxn.dat"
        with open(path_rxn_E,'r') as f:
            rxn_list = [line.split()[0].split("/")
                        for line in f.readlines()[1:]
                        if 'needs' not in line]

        n_rxn = len(rxn_list)
        print(f"{ion_type}, {ion_energy}, Number of reactions: {n_rxn}")

        path_gas = f"{src}/post_process_bulk_gas/gas"
        ec = EnergyCalculator(src, path_gas)
        E_1, E_2 = np.zeros(n_rxn), np.zeros(n_rxn)
        for i in range(n_rxn):
            rxn_info_before, rxn_info_after = rxn_list[i]
            rxn_info_before = rxn_info_before.split(",")
            rxn_info_after = rxn_info_after.split(",")

            try:
                E_before_1 = ec.run(rxn_info_before, self.cal_type_1)
                E_before_2 = ec.run(rxn_info_before, self.cal_type_2)
                E_after_1 = ec.run(rxn_info_after, self.cal_type_1)
                E_after_2 = ec.run(rxn_info_after, self.cal_type_2)
            except EnergyNotCalculatedError as e:
                print(f"Error: {e}")
                continue

            E_1[i] = E_after_1 - E_before_1
            E_2[i] = E_after_2 - E_before_2

        data = [E_1, E_2]
        return data

class EnergyCalculator:
    def __init__(self, src, path_gas):
        self.src = src
        self.path_gas = path_gas

    def run(self, rxn_info, cal_type):
        if ',' in rxn_info:
            idx, gas_list = rxn_info.split(",")
        else:
            idx, gas_list = rxn_info[0], []

        E_slab = self.get_slab_energy(idx, cal_type)
        E_gas = self.get_gas_energy(gas_list, cal_type)
        E_rxn = E_slab + E_gas
        return E_rxn

    def get_slab_energy(self, idx, cal_type):
        incidence = idx.split("_")[0]
        path_slab_E = f"{self.src}/post_process_bulk_gas/{incidence}/{idx}/{cal_type}/e"
        if not os.path.exists(path_slab_E):
            raise EnergyNotCalculatedError(f"Energy of {idx} not calculated")
        with open(path_slab_E, 'r') as f:
            line = f.readline()
            E_slab = float(line.split()[-1])
        return E_slab

    def get_gas_energy(self, gas_list, cal_type):
        E_gas = 0.0
        if not gas_list:
            return 0
        for gas_type in gas_list:
            idx = gas_type.split("_")
            path_gas_E = f"{self.path_gas}/{idx}/{cal_type}/e"
            if not os.path.exists(path_gas_E):
                raise EnergyNotCalculatedError(f"Energy of {gas_type} not calculated")
            with open(path_gas_E, 'r') as f:
                E_gas += float(f.readline().split()[-1])
        return E_gas

class DataProcessor:
    def __init__(self, src, cal_type_1, cal_type_2):
        self.src = src
        self.cal_type_1 = cal_type_1
        self.cal_type_2 = cal_type_2
        self.ion_types = ['CF', 'CF3', 'CH2F']
        self.ion_energies = [20, 50]

    @pklSaver.run(lambda self: f"rxn_E_total_{self.cal_type_1}_{self.cal_type_2}.pkl")
    def run(self):
        data_total = []
        dl = DataLoader(self.src, self.cal_type_1, self.cal_type_2)
        for ion_type in self.ion_types:
            for ion_E in self.ion_energies:
                data = dl.run(ion_type, ion_E)
                if not data_total:
                    data_total = data
                else:
                    data_total = [np.concatenate((data_total[i], data[i])) for i in range(2)]
        return data_total

class DataPlotter:
    def __init__(self, cal_type_1, cal_type_2):
        self.cal_type_1 = cal_type_1
        self.cal_type_2 = cal_type_2
        self.fig_title = "Total"

    def run(self, data):
        print("plotting...", end='')
        fig, ax = self.generate_figure()
        rxn_1, rxn_2, Z = self.sort_data(data)
        self.plot(rxn_1, rxn_2, Z, ax)
        self.add_statistics(data, ax)
        self.set_xy_limits(ax, rxn_1, rxn_2)
        self.decorate(data, ax)
        self.save(fig)
        print("done")

    def generate_figure(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(2.5,2)
        ax.set_aspect('equal','box')
        return fig, ax

    def plot(self, rxn_1, rxn_2, Z, ax):
        cmap = plt.get_cmap('Blues')
        norm = matplotlib.colors.LogNorm(vmin=1)
        plot_options = {
            'c': Z+1,
            'cmap': cmap,
            's': 1,
            'norm': norm,
            'zorder': 2,
            }
        ax.scatter(rxn_1, rxn_2, **plot_options)

    def sort_data(self, data):
        E_1, E_2 = data
        E_1, E_2 = np.array(E_1), np.array(E_2)
        values = np.vstack((E_1, E_2))
        kernel = gaussian_kde(values)
        kernel.set_bandwidth(bw_method=kernel.factor*3)
        Z = kernel(values)
        Z /= np.min(Z[np.nonzero(Z)])
        idx = Z.argsort()
        DFT_rxn, NNP_rxn, Z = E_1[idx], E_2[idx], Z[idx]

        return DFT_rxn, NNP_rxn, Z

    def add_statistics(self, data, ax):
        E_1, E_2 = data
        mae = np.mean(np.abs(E_1-E_2))
        pearson = np.corrcoef(E_1,E_2)[0,1]
        r2 = 1-np.sum(np.power(E_1-E_2,2))/np.sum(np.power(E_1-np.mean(E_1),2))
        text = f"MAE: {mae:.2f} eV\nPearson: {pearson:.4f}\nR2: {r2:.4f}"
        box_options = {
            'transform': ax.transAxes,
            'ha': 'left',
            'va': 'top',
            'bbox': {
                'boxstyle': 'round',
                'facecolor': 'wheat',
                'alpha': 0.5,
                },
            'fontsize': 5,
        }
        ax.text(0.05, 0.95, text, **box_options)

    def set_xy_limits(self, ax, rxn_1, rxn_2):
        x_min = np.min([np.min(rxn_1), np.min(rxn_2)])
        x_max = np.max([np.max(rxn_1), np.max(rxn_2)])
        if x_min < 0:
            x_min *= 1.1
        else:
            x_min *= 0.9
        if x_max < 0:
            x_max *= 0.9
        else:
            x_max *= 1.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)

    def decorate(self, data, ax):
        ax.axline((0, 0), slope=1, linestyle='--', zorder=1, color='grey')
        ax.set_aspect('equal')

        cal_type_1 = self.cal_type_1.upper()
        cal_type_2 = self.cal_type_2.upper()
        x_label = fr"$E^{{\rm{{{cal_type_1}}}}}_{{\rm{{rxn}}}}$ (eV)"
        y_label = fr"$E^{{\rm{{{cal_type_2}}}}}_{{\rm{{rxn}}}}$ (eV)"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        E_1, _ = data
        n_data = len(E_1)
        title = f"{self.fig_title} ({n_data} rxns)"
        ax.set_title(title)

    def save(self, fig):
        save_options = {
            'dpi': 400,
            'bbox_inches': 'tight',
        }
        path_save = f"rxn_E_{self.fig_title}_{self.cal_type_1}_{self.cal_type_2}.png"
        fig.savefig(path_save, **save_options)

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot.py input.yaml")
        sys.exit(1)

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        inputs = yaml.safe_load(f)
    src = inputs['src']
    cal_type_1 = inputs['cal_type_1']
    cal_type_2 = inputs['cal_type_2']

    dp = DataProcessor(src, cal_type_1, cal_type_2)
    data = dp.run()

    pl = DataPlotter(cal_type_1, cal_type_2)
    pl.run(data)


if __name__ == "__main__":
    main()
