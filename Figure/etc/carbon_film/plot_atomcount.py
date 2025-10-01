import os
import sys
import time
from functools import wraps
import pickle
from dataclasses import dataclass
import string

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


@dataclass
class PLOT_OPTS:
    EXCLUDE_LIST = [
        'NOT_CREATED',
        'REFLECTED',
        'REMOVED_DURING_MD',
        # 'etc',
        # 'BYPRODUCT',
    ]

    STYLE_LIST = {
        'BYPRODUCT': ('skyblue', '-'),
        'SiC_cluster': ('orange', '-'),
        'SiC_cluster_with_F': ('orange', 'dashed'),
        'SiC_cluster_with_F': ('orange', 'dotted'),

        'Fluorocarbon': ('green', '-'),
        'Fluorocarbon_with_O': ('green', 'dotted'),

        'C4': ('pink', '-'),
        'C3': ('red', '-'),
        'C2': ('blue', '-'),
        'etc': ('purple', '-'),

        'CX': ('red', '-'),
    }

    GLOBAL_LEGEND_OPTS = {
        'loc': 'upper right',
        'bbox_to_anchor': (0.9, 0.9),
        'ncol': 1,
        'fontsize': 28,
    }

    GLOBAL_LEGEND_FLAT_OPTS = {
        'loc': 'lower center',
        'bbox_to_anchor': (0.5, 0.0),
        'ncol': 4,
        'fontsize': 20,
    }

    GLOBAL_LEGEND_COMPARE_OPTS = {
        'loc': 'lower center',
        'bbox_to_anchor': (0.5, 0.0),
        'ncol': 4,
        'fontsize': 18,
    }

    ION_NAME_CONVERT = {
        'CF': 'CF$^{+}$',
        'CF2': 'CF${}_{2}^{+}$',
        'CF3': 'CF${}_{3}^{+}$',
        'CH2F': 'CH$_2$F$^{+}$',
        'CHF2': 'CHF${}_{2}^{+}$',
    }


def timeit(function):
    '''
    Wrapper function to measure the execution time of a function.
    '''
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print(f'{function.__name__:40s} took {end - start:10.4f} seconds')
        return result
    return wrapper


class DataLoader():
    @timeit
    @staticmethod
    def get_data(src, name):
        path_pkl = f'{name}.pkl'
        if os.path.exists(path_pkl):
            with open(path_pkl, 'rb') as f:
                data = pickle.load(f)
            return data

        df = pd.read_hdf(src, key='df')
        # df = pd.read_hdf(src)
        data = {}
        for row in df.itertuples(index=False):
            s_idx = int(row.struct_idx)
            state_C = row.state_C
            if state_C not in data:
                data[state_C] = {}
            if s_idx not in data[state_C]:
                data[state_C][s_idx] = 0
            data[state_C][s_idx] += 1

        with open(path_pkl, 'wb') as f:
            pickle.dump(data, f)
        return data

class DataPlotter():
    def plot(self, data_dict):
        ions, ion_Es = self.get_ion_types_and_energies(data_dict)
        n_row, n_col = len(ion_Es), len(ions)
        fig, axes = self.generate_figure(n_row, n_col)
        used_axes = set()
        for name, data in data_dict.items():
            ion, ion_E = name.split('_')
            ax = self.select_ax(axes, used_axes, ions, ion_Es, ion, ion_E)
            self.plot_each_state(data, ax)
            title = f'{PLOT_OPTS.ION_NAME_CONVERT[ion]} {ion_E}'
            self.decorate_axes(ax)
            ax.set_title(title)
        self.remove_unused_axes(fig, n_row, n_col, axes, used_axes)
        self.add_global_legend(fig, axes, PLOT_OPTS.GLOBAL_LEGEND_OPTS)
        fig.tight_layout()
        self.save(fig)

    def plot_flatten(self, data_dict, n_row, n_col):
        fig, axes = self.generate_figure(n_row, n_col)
        ax_dict = {}
        for name, ax in zip(data_dict.keys(), axes.flat):
            ax_dict[name] = ax

        for (name, data), idx_abc in zip(data_dict.items(), string.ascii_lowercase):
            ion, ion_E = name.split('_')
            ax = ax_dict[name]
            self.plot_each_state(data, ax)
            title = f'({idx_abc}) {PLOT_OPTS.ION_NAME_CONVERT[ion]} {ion_E}'
            self.decorate_axes(ax)
            ax.text(-0.15, 1.2, title, transform=ax.transAxes,
                    fontsize=18, va='top', ha='left')

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        self.add_global_legend(fig, axes, PLOT_OPTS.GLOBAL_LEGEND_FLAT_OPTS)
        self.save(fig)

    def plot_compare_two(self, data_dict):
        fig, axes = self.generate_figure(1, 2)
        ax_dict = {}
        for name, ax in zip(data_dict.keys(), axes.flat):
            ax_dict[name] = ax

        for (name, data), idx_abc in zip(data_dict.items(), string.ascii_lowercase):
            ion, ion_E = name.split('_')
            ax = ax_dict[name]
            self.plot_each_state(data, ax)
            title = f'({idx_abc}) {PLOT_OPTS.ION_NAME_CONVERT[ion]} {ion_E}'
            self.decorate_axes(ax)
            ax.text(-0.15, 1.2, title, transform=ax.transAxes,
                    fontsize=18, va='top', ha='left')

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.3)
        self.add_global_legend(fig, axes, PLOT_OPTS.GLOBAL_LEGEND_COMPARE_OPTS)
        self.save(fig)

    def plot_each_state(self, data, ax):
        for state_C in sorted(data.keys()):
            d = data[state_C]
            if state_C in PLOT_OPTS.EXCLUDE_LIST:
                continue
            x = np.array(list(d.keys()))
            y = np.array(list(d.values()))
            if np.max(y) < 10:
                continue
            color, linestyle = PLOT_OPTS.STYLE_LIST.get(state_C, ('grey', '-'))
            ax.plot(x, y, label=state_C, color=color, linestyle=linestyle,
                    linewidth=2, alpha=0.6)

    def select_ax(self, axes, used_axes, ions, ion_Es, ion, ion_E):
        idx_row, idx_col = ion_Es[ion_E], ions[ion]
        ax = axes[idx_row][idx_col]
        used_axes.add((idx_row, idx_col))
        return ax

    def save(self, fig):
        fig.savefig('atomcount.png', dpi=300)
        fig.savefig('atomcount.pdf', dpi=300)
        fig.savefig('atomcount.eps', dpi=300)

    def get_ion_types_and_energies(self, data_dict):
        keys = [i for i in data_dict.keys()]
        ions = sorted(list(set([i.split('_')[0] for i in keys])))
        ions = {ion: idx for idx, ion in enumerate(ions)}
        ion_Es = sorted(set([i.split('_')[1] for i in keys]), key=lambda x: int(x.replace('eV', '')))
        ion_Es = {ion_E: idx for idx, ion_E in enumerate(ion_Es)}
        return ions, ion_Es

    def generate_figure(self, n_row, n_col):
        mpl.rcParams['font.family'] = 'Arial'
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(n_row, n_col, figsize=(1 + n_col * 4, 1 + n_row * 4), squeeze=False)
        return fig, axes

    def remove_unused_axes(self, fig, n_row, n_col, axes, used_axes):
        for i in range(n_row):
            for j in range(n_col):
                if (i, j) not in used_axes:
                    fig.delaxes(axes[i][j])

    def decorate_axes(self, ax):
        ax.set_xlabel('Structure index')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 9000)
        # ax.set_xlim(0, 7000)
        ax.set_ylim(0, 2000)

    def add_global_legend(self, fig, axes, legend_opts):
        handle_labels = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in handle_labels:
                    handle_labels[label] = handle
        fig.legend(handle_labels.values(), handle_labels.keys(), **legend_opts)


def main():
    if len(sys.argv) != 2:
        print('Usage: python plot_atomcount.py <path_yaml>')
        sys.exit(1)

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        config = yaml.safe_load(f)

    data_dict = {}
    for ion, energies in config.items():
        for ion_E, path in energies.items():
            print(ion, ion_E, path)
            if isinstance(ion_E, str) and '_' in ion_E:
                ion_E, text = ion_E.split('_')
                name = f"{ion}_{ion_E}eV, {text}"
            else:
                name = f"{ion}_{ion_E}eV"
            src = f'{path}/total_dict.h5'
            data = DataLoader.get_data(src, name)
            data_dict[name] = data
            print(f'Processed {name}')

    dp = DataPlotter()
    # dp.plot(data_dict)
    dp.plot_flatten(data_dict, 3, 3)
    # dp.plot_compare_two(data_dict)


if __name__ == "__main__":
    main()
