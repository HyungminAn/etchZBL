import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import yaml

from utils import PARAMS, pklSaver


class DataLoader:
    def __init__(self, src, cal_type_1, cal_type_2, log_dict):
        self.src = src
        self.cal_type_1 = cal_type_1
        self.cal_type_2 = cal_type_2
        self.log_dict = log_dict

    def run(self, ion_type, ion_energy):
        self.log(f"Processing {ion_type} at {ion_energy} eV", self.log_dict['log'])
        title = f"{ion_type} at {ion_energy} eV"
        self.log(f"{title:*^80}", self.log_dict['energy'])
        self.log(f"{title:*^80}", self.log_dict['need_change'])

        src = f"{self.src}/{ion_type}/{ion_energy}"
        rxn_list = self.get_reaction_list(src)
        if rxn_list is None:
            self.log(f"No reactions found for {ion_type} at {ion_energy} eV", self.log_dict['log'])
            return None

        n_rxn = len(rxn_list)
        self.log(f"{ion_type}, {ion_energy}, Number of reactions: {n_rxn}", self.log_dict['log'])
        line = [f"E_before ({self.cal_type_1})",
                f"E_after ({self.cal_type_1})",
                f"E_before ({self.cal_type_2})",
                f"E_after ({self.cal_type_2})",
                f"E_rxn ({self.cal_type_1})",
                f"E_rxn ({self.cal_type_2})"]
        line = [f"{item:>20}" for item in line]
        self.log(f"{''.join(line)}", self.log_dict['energy'])

        path_gas = f"{src}/{PARAMS.path_post_process}/gas"
        ec = EnergyCalculator(src, path_gas, self.log_dict)
        E_1, E_2 = {}, {}
        for i in range(n_rxn):
            self.log(f"Processing reaction {i+1}/{n_rxn}", self.log_dict['log'])
            rxn_info_before, rxn_info_after = rxn_list[i]
            rxn_info_before = rxn_info_before.split(",")
            rxn_info_after = rxn_info_after.split(",")

            E_before_1 = ec.run(rxn_info_before, self.cal_type_1)
            E_before_2 = ec.run(rxn_info_before, self.cal_type_2)
            E_after_1 = ec.run(rxn_info_after, self.cal_type_1)
            E_after_2 = ec.run(rxn_info_after, self.cal_type_2)

            if E_before_1 is not None and E_after_1 is not None:
                E_rxn_1 = E_after_1 - E_before_1
            else:
                E_rxn_1 = None
            if E_before_2 is not None and E_after_2 is not None:
                E_rxn_2 = E_after_2 - E_before_2
            else:
                E_rxn_2 = None

            if E_rxn_1 is not None and E_rxn_2 is not None:
                E_1[i] = E_rxn_1
                E_2[i] = E_rxn_2

            line_energy = ""
            for item in [E_before_1, E_after_1, E_before_2, E_after_2, E_rxn_1, E_rxn_2]:
                if item is not None:
                    line_energy += f"{item:20.2f}"
                else:
                    line_energy += f"{'-':>20}"
            self.log(line_energy, self.log_dict['energy'])

        E_1 = np.array([v for v in E_1.values() if v is not None])
        E_2 = np.array([v for v in E_2.values() if v is not None])
        data = [E_1, E_2]
        return data

    def get_reaction_list(self, src):
        path_rxn_E = f"{src}/{PARAMS.path_reaction_data}"
        if not os.path.exists(path_rxn_E):
            return None

        with open(path_rxn_E, 'r') as f:
            lines = f.readlines()[1:]
            rxn_list = [line.split()[0].split("/")
                        for line in lines
                        if 'needs' not in line]
            unprocessed_lines = [line for line in lines if 'needs' in line]
            for line in unprocessed_lines:
                self.log(line, self.log_dict['need_change'])

        return rxn_list

    def log(self, message, logfile):
        with open(logfile, 'a') as f:
            f.write(message + "\n")
        print(message)

class EnergyCalculator:
    def __init__(self, src, path_gas, log_dict):
        self.src = src
        self.path_gas = path_gas
        self.log_dict = log_dict

    def run(self, rxn_info, cal_type):
        if len(rxn_info) > 1:
            idx, gas_list = rxn_info[0], rxn_info[1:]
        else:
            idx, gas_list = rxn_info[0], []

        E_slab = self.get_slab_energy(idx, cal_type)
        E_gas = self.get_gas_energy(gas_list, cal_type)
        if E_slab is None or E_gas is None:
            self.log(f"Energy for {idx} not calculated", self.log_dict['log'])
            return None

        E_rxn = E_slab + E_gas
        return E_rxn

    def get_slab_energy(self, idx, cal_type):
        incidence = idx.split("_")[0]
        path_slab_E = f"{self.src}/{PARAMS.path_post_process}/{incidence}/{idx}/{cal_type}/e"
        if not os.path.exists(path_slab_E):
            self.log(f"Energy of slab {idx} not calculated", self.log_dict['log'])
            self.log(f"{path_slab_E}", self.log_dict['unfinished'])
            return None

        with open(path_slab_E, 'r') as f:
            line = f.readline()
            E_slab = float(line.split()[-1])
        return E_slab

    def get_gas_energy(self, gas_list, cal_type):
        E_gas = 0.0
        if not gas_list:
            return E_gas

        for gas_type in gas_list:
            # idx = gas_type.split("_")
            # path_gas_E = f"{self.path_gas}/{idx}/{cal_type}/e"
            path_gas_E = f"{self.path_gas}/{gas_type}/{cal_type}/e"
            if not os.path.exists(path_gas_E):
                self.log(f"Energy of gas {gas_type} not calculated", self.log_dict['log'])
                self.log(f"{path_gas_E}", self.log_dict['unfinished'])
                return None

            with open(path_gas_E, 'r') as f:
                E_gas += float(f.readline().split()[-1])
        return E_gas

    def log(self, message, logfile):
        with open(logfile, 'a') as f:
            f.write(message + "\n")
        print(message)

class DataProcessor:
    def __init__(self, src, cal_type_1, cal_type_2):
        self.src = src
        self.cal_type_1 = cal_type_1
        self.cal_type_2 = cal_type_2
        self.log_dict = {
                'log': 'LOG',
                'energy': 'LOG_energy',
                'unfinished': 'LOG_unfinished',
                'need_change': 'LOG_need_change',
                }

    # @pklSaver.run(lambda self: f"rxn_E_total_{self.cal_type_1}_{self.cal_type_2}.pkl")
    def run(self):
        data_total = []
        self.initialize_log()
        dl = DataLoader(self.src,
                        self.cal_type_1,
                        self.cal_type_2,
                        self.log_dict)
        for ion_type in PARAMS.DUMP_INFO.ions:
            for ion_E in PARAMS.DUMP_INFO.energies:
                data = dl.run(ion_type, ion_E)
                if data is None:
                    continue

                if not data_total:
                    data_total = data
                else:
                    data_total = [np.concatenate((data_total[i], data[i])) for i in range(2)]
        self.remove_duplicates_in_log()
        return data_total

    def initialize_log(self):
        for path_log in self.log_dict.values():
            with open(path_log, 'w') as f:
                f.write("")

    def log(self, logfile, message):
        with open(logfile, 'a') as f:
            f.write(message + "\n")
        print(message)

    def remove_duplicates_in_log(self):
        file = self.log_dict['unfinished']
        with open(file, 'r') as f:
            lines = f.readlines()
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
        with open(file, 'w') as f:
            f.writelines(unique_lines)

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
