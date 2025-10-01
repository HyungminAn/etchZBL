import os
import pickle
from itertools import product
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error as mae

from params import PARAMS

class FigureGenerator:
    def run(self):
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 14})
        n_row = len(PARAMS.energy_list)
        n_col = len(PARAMS.ion_list)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4))
        return fig, axes

class AbstractPlotter(ABC):
    def run(self, data):
        fg = FigureGenerator()
        fig, axes = fg.run()
        self.plot(data, axes)
        self.decorate(fig, axes)
        self.save(fig)

    @abstractmethod
    def plot(self, data, axes):
        pass

    @abstractmethod
    def decorate(self, fig, axes):
        pass

    @abstractmethod
    def save(self, fig):
        pass

class EnergyPlotter(AbstractPlotter):
    def plot(self, data, axes):
        label_count = {}
        for (pot, ion, energy), thermo in data.items():
            row = PARAMS.energy_list.index(energy)
            col = PARAMS.ion_list.index(ion)
            ax = axes[row, col]
            ax.set_title(f"{PARAMS.ION_CONVERT_DICT[ion]}, {energy}eV")
            style = PARAMS.STYLE_DICT[pot]

            time = thermo[:, 1]
            pot_e = thermo[:, 3]

            label = PARAMS.POT_CONVERT_DICT[pot]
            if label not in label_count:
                label_count[label] = True
            else:
                label = None
            ax.plot(time, pot_e, label=label, **style)

    def decorate(self, fig, axes):
        for ax in axes.flat:
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Potential Energy (eV)")
            ax.set_xlim(0, None)

        self.set_global_legend(fig, axes)

    def set_global_legend(self, fig, axes):
        handles = []
        labels = []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))

        fig.legend(
            handles, labels,
            loc='lower center',
            frameon=False,
            ncol=2,
            bbox_to_anchor=(0.5, 0.02)
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)

    def save(self, fig):
        fig.savefig("energy_plot.png")

class AtomCountPlotter(AbstractPlotter):
    def plot(self, data, axes):
        label_count = {}
        for (pot, ion, energy), thermo in data.items():
            row = PARAMS.energy_list.index(energy)
            col = PARAMS.ion_list.index(ion)
            ax = axes[row, col]
            ax.set_title(f"{PARAMS.ION_CONVERT_DICT[ion]}, {energy}eV")
            style = PARAMS.STYLE_DICT[pot]

            time = thermo[:, 1]
            atom_numbers = {
                    'Si': thermo[:, -5],
                    'O': thermo[:, -4],
                    'C': thermo[:, -3],
                    'H': thermo[:, -2],
                    'F': thermo[:, -1],
                    }

            for atom, counts in atom_numbers.items():
                color = PARAMS.COLOR_DICT.get(atom)
                label = f"{atom} ({PARAMS.POT_CONVERT_DICT[pot]})"
                if label not in label_count:
                    label_count[label] = True
                else:
                    label = None
                ax.plot(time, counts, label=label, color=color, **style)

    def decorate(self, fig, axes):
        for ax in axes.flat:
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Atom Count")
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)
        self.set_global_legend(fig, axes)

    def set_global_legend(self, fig, axes):
        handles = []
        labels = []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))

        fig.legend(
            handles, labels,
            loc='lower center',
            frameon=False,
            ncol=5,
            bbox_to_anchor=(0.5, 0.02)
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)

    def save(self, fig):
        fig.savefig("atom_counts_plot.png")

class DataLoader:
    def __init__(self):
        self.path_save = 'data_EF.pkl'
        self.path_log_energy = 'log_energy.txt'
        self.path_log_force = 'data_force.pkl'
        self.path_time_dict = 'log_time.txt'

    def run(self, data_ref, data):
        if os.path.exists(self.path_save):
            with open(self.path_save, 'rb') as f:
                return pickle.load(f)

        pot_type = PARAMS.pot_type
        ion_list = PARAMS.ion_list
        energy_list = PARAMS.energy_list
        incidences = PARAMS.incidences
        img_idxes = [i for i in range(PARAMS.n_max_sample)]

        time_dict = self.get_time_dict()
        result = {'energy': {}, 'force': {}}
        log_save = {}
        for pot, ion, ion_energy, inc, idx in product(pot_type, ion_list, energy_list, incidences, img_idxes):
            key = (pot, ion, ion_energy)
            if key not in result['energy']:
                result['energy'][key] = {
                        'time': [],
                        'xy': [],
                        'energy_diff': []
                        }
            if key not in result['force']:
                result['force'][key] = {
                        'force_x': [],
                        'force_y': []
                        }

            key_long = (pot, ion, ion_energy, inc, idx)
            pot_e_ref, force_ref = data_ref.get(key_long, (None, None))
            pot_e, force = data.get(key_long, (None, None))
            time = time_dict.get(key_long, None)
            if (pot_e_ref is None or force_ref is None or pot_e is None or force is None or time is None):
                continue
            nions, nions_ref = len(force), len(force_ref)
            pot_e_diff = (pot_e - pot_e_ref) / nions * 1000  # Convert to meV/atom

            result['energy'][key]['time'].append(time)
            result['energy'][key]['xy'].append((pot_e_ref / nions, pot_e / nions))
            result['energy'][key]['energy_diff'].append(pot_e_diff)
            print(f"Processed: {pot}, {ion}, {ion_energy}, {inc}, {idx}")

            result['force'][key]['force_x'].append(force_ref)
            result['force'][key]['force_y'].append(force)

            log_save[key_long] = (time, pot_e_ref, pot_e, pot_e_diff)

        with open(self.path_save, 'wb') as f:
            pickle.dump(result, f)

        with open(self.path_log_energy, 'w') as f:
            f.write((f"{'pot':>8} {'ion':>5} {'ionE':>5} {'inc':>5} {'idx':>5} "
                     f"{'time':>8} {'E_ref':>8} {'E':>8} {'E_diff':>8}\n"))
            for (pot, ion, ion_energy, inc, idx), (time, pot_e_ref, pot_e, pot_e_diff) in log_save.items():
                f.write(f"{pot:>8} {ion:>5} {ion_energy:>5} {inc:>5} {idx:>5} ")
                f.write(f"{time:>8.2f} {pot_e_ref:>8.2f} {pot_e:>8.2f} {pot_e_diff:>8.2f}\n")

        return result

    def get_time_dict(self):
        with open(self.path_time_dict, 'r') as f:
            lines = f.readlines()
        time_dict = {}
        for line in lines[1:]:
            pot, ion, ion_energy, inc, idx, time = line.strip().split()
            ion_energy = int(ion_energy)
            inc = int(inc)
            idx = int(idx)
            time = float(time)

            key = (pot, ion, ion_energy, inc, idx)
            time_dict[key] = time
        return time_dict

class EnergyDiffPlotter(AbstractPlotter):
    def plot(self, data, axes):
        data_x, data_y = data['x'], data['y']
        dl = DataLoader()
        plot_values = dl.run(data_x, data_y)['energy']
        for (pot, ion, ion_energy), values in plot_values.items():
            row = PARAMS.energy_list.index(ion_energy)
            col = PARAMS.ion_list.index(ion)
            ax = axes[row, col]
            ax.set_title(f"{PARAMS.ION_CONVERT_DICT[ion]}, {ion_energy}eV")
            style = PARAMS.STYLE_DICT[pot]

            time = values['time']
            pot_e_diff = values['energy_diff']

            label = PARAMS.POT_CONVERT_DICT[pot]
            ax.plot(time, pot_e_diff, label=label, **style)

    def decorate(self, fig, axes):
        for ax in axes.flat:
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Energy diff (meV/atom)")
            ax.set_xlim(0, None)
            ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        self.set_global_legend(fig, axes)

    def set_global_legend(self, fig, axes):
        handles = []
        labels = []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
        handles, labels = zip(*sorted(zip(unique_handles, unique_labels), key=lambda x: x[1]))

        fig.legend(
            handles, labels,
            loc='lower center',
            frameon=False,
            ncol=2,
            bbox_to_anchor=(0.5, 0.02)
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)

    def save(self, fig):
        fig.savefig("energy.png")

class ParityPlotter:
    def generate_figure(self):
        pot_type = PARAMS.pot_type
        ion_list = PARAMS.ion_list
        energy_list = PARAMS.energy_list
        n_row = len(pot_type) * 2
        n_col = len(energy_list) * len(ion_list)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4))
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 14})

        ax_dict = {}
        for (pot, ion, ion_e) in product(pot_type, ion_list, energy_list):
            row = pot_type.index(pot) * 2
            col = energy_list.index(ion_e) + ion_list.index(ion) * len(energy_list)
            key = (pot, ion, ion_e)
            ax_dict[key] = axes[row, col], axes[row + 1, col]
        return fig, ax_dict

    def run(self, data):
        fig, ax_dict = self.generate_figure()
        energies, forces = self.gather_data(data)
        for key, (ax_energy, ax_force) in ax_dict.items():
            self.plot_energy(energies[key], ax_energy, key)
            self.plot_force(forces[key], ax_force, key)
        # self.decorate(fig, ax_dict)
        self.save(fig)

    def gather_data(self, data):
        dl = DataLoader()
        data = dl.run(data['x'], data['y'])
        energies = {}
        for key, values in data['energy'].items():
            if key not in energies:
                energies[key] = {'x': [], 'y': []}
            energies[key]['x'].extend([i[0] for i in values['xy']])
            energies[key]['y'].extend([i[1] for i in values['xy']])
        forces = {}
        for key, values in data['force'].items():
            if key not in forces:
                forces[key] = {'x': [], 'y': []}
            forces[key]['x'].extend([i.flatten() for i in values['force_x']])
            forces[key]['y'].extend([i.flatten() for i in values['force_y']])
        forces_merged = {}
        for key in forces:
            x = np.concatenate(forces[key]['x'])
            y = np.concatenate(forces[key]['y'])
            forces_merged[key] = {'x': x, 'y': y}
        return energies, forces_merged

    def plot_energy(self, data, ax, key):
        x, y = data['x'], data['y']
        c = PARAMS.parity_plot_color_dict[key[0]]
        ax.scatter(x, y, s=1, alpha=0.5, color=c)
        ax.axline((0, 0), slope=1, linestyle='--', color='grey', alpha=0.5)

        max_val = max(max(x), max(y))
        min_val = min(min(x), min(y))
        interval = max_val - min_val
        ax.set_xlim(min_val - interval * 0.05, max_val + interval * 0.05)
        ax.set_ylim(min_val - interval * 0.05, max_val + interval * 0.05)
        ax.set_aspect('equal')
        ax.set_xlabel(r"$E_{\mathrm{NNP}}^{\mathrm{ref}}$ (eV/atom)")
        ax.set_ylabel(r"$E_{\mathrm{NNP}}^{\mathrm{test}}$ (eV/atom)")
        r2_score = r2(x, y)
        mae_score = mae(x, y)
        text = f"R$^2$: {r2_score:.3f}\nMAE: {mae_score:.3f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha='left', va='top',)

        pot, ion, ion_e = key
        title = f'{PARAMS.POT_CONVERT_DICT[pot]}, {PARAMS.ION_CONVERT_DICT[ion]}, {ion_e}eV'
        ax.set_title(title)

    def plot_force(self, data, ax, key):
        x, y = data['x'], data['y']
        c = PARAMS.parity_plot_color_dict[key[0]]
        ax.scatter(x, y, s=1, alpha=0.5, color=c)
        ax.axline((0, 0), slope=1, linestyle='--', color='grey', alpha=0.5)

        max_val = max(max(x), max(y))
        min_val = min(min(x), min(y))
        interval = max_val - min_val
        ax.set_xlim(min_val - interval * 0.05, max_val + interval * 0.05)
        ax.set_ylim(min_val - interval * 0.05, max_val + interval * 0.05)
        ax.set_aspect('equal')
        ax.set_xlabel(r"$F_{\mathrm{NNP}}^{\mathrm{ref}}$ (eV/$\mathrm{\AA}$)")
        ax.set_ylabel(r"$F_{\mathrm{NNP}}^{\mathrm{test}}$ (eV/$\mathrm{\AA}$)")
        r2_score = r2(x, y)
        mae_score = mae(x, y)
        text = f"R$^2$: {r2_score:.3f}\nMAE: {mae_score:.3f}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha='left', va='top',)

    def decorate(self, fig, axes):
        pass

    def save(self, fig):
        fig.tight_layout()
        fig.savefig("parity_plot.png")
