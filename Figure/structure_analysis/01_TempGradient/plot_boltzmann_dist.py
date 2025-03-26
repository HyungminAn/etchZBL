import os
import sys
from dataclasses import dataclass
import pickle

from matplotlib import pyplot as plt
import numpy as np
from ase.calculators.lammps import convert
from ase.io import read

def print_process(func):
    def wrapper(*args, **kwargs):
        print(f"Running function: {func.__name__}")
        result = func(*args, **kwargs)
        print("Done")
        return result
    return wrapper

def cache_with_pickle(pickle_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check if the pickle file exists
            if os.path.exists(pickle_path):
                print(f"Loading cached result from {pickle_path}...")
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)

            # If not, execute the function body
            print(f"Cache not found. Executing function {func.__name__}...")
            result = func(*args, **kwargs)

            # Save the result to the pickle file
            print(f"Saving result to {pickle_path}...")
            with open(pickle_path, 'wb') as f:
                pickle.dump(result, f)

            return result
        return wrapper
    return decorator


@dataclass
class PLOT_OPTS:
    MASS = {
            'Si': 28.0855,
            'O': 15.999,
            'C': 12.011,
            'F': 18.998,
            }
    COLOR = {
            'Si': '#ffedb7',
            'O': '#ff0d0d',
            'C': '#909090',
            'F': '#ff55ff',
            }
    READ = {
            'format': 'lammps-dump-text',
            }
@dataclass
class CONSTANTS:
    AMU_TO_KG = 1.66053906660e-27  # kg
    T = 300  # K
    K = 1.38e-23  # J/K

class DumpReader:
    #@cache_with_pickle('data.pickle')
    @print_process
    @staticmethod
    def read_data(path_dump, idx):
        if idx is not None:
            dump = read(path_dump, index=idx, **PLOT_OPTS.READ)
            dump = [dump]
        else:
            dump = read(path_dump, index=':', **PLOT_OPTS.READ)

        result = {idx: DumpReader.get_speed_from_image(image) for idx, image in enumerate(dump)}
        return result, dump

    @staticmethod
    def gen_mask(image, symbol):
        fix_h = 6.0
        mask = np.array([
            idx for idx, atom in enumerate(image)
            if atom.symbol == symbol and atom.position[2] > fix_h])
        return mask

    @staticmethod
    def get_speed_from_image(image):
        MPS_TO_ANGSTROM_PS = 1e-2
        VEL_CONV_FACTOR = convert(1.0, 'velocity', 'ASE', 'metal') / MPS_TO_ANGSTROM_PS
        sub_dict = {}
        velocities = image.get_velocities() * VEL_CONV_FACTOR
        for symbol in PLOT_OPTS.MASS.keys():
            mask = DumpReader.gen_mask(image, symbol)
            speeds = np.linalg.norm(velocities[mask], axis=1)
            sub_dict[symbol] = (mask, speeds)
        return sub_dict


class TemperaturePlotter:
    @print_process
    @staticmethod
    def plot(data, temp, output):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        ax_dict = {
                'Si': axes[0, 0],
                'O': axes[0, 1],
                'C': axes[1, 0],
                'F': axes[1, 1],
                }
        TemperaturePlotter.plot_hist(data, ax_dict)
        TemperaturePlotter.plot_boltzmann_dist(ax_dict, temp)

        fig.suptitle(f'{output} / Average tmp: {temp:.2f} K')
        fig.tight_layout()
        fig.savefig(f'{output}.png')

    @staticmethod
    def plot_hist(data, ax_dict):
        for symbol, (_, speeds) in data.items():
            plot_opts = {
                    'alpha': 0.5,
                    'color': PLOT_OPTS.COLOR[symbol],
                    'label': f'{symbol} speeds',
                    }
            ax = ax_dict[symbol]
            ax.hist(speeds, bins=50, density=True, **plot_opts)
            ax.set_xlabel('Speed [m/s]')
            ax.set_ylabel('Probability density')
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)

    @staticmethod
    def plot_boltzmann_dist(ax_dict, temp):
        mass_dict_in_kg = {key: value * CONSTANTS.AMU_TO_KG for key, value in PLOT_OPTS.MASS.items()}
        for key, value in mass_dict_in_kg.items():
            ax = ax_dict[key]
            plot_opts = {
                    'label': f'{key} Boltzmann distribution',
                    'color': PLOT_OPTS.COLOR[key],
                    }
            v_min, v_max = ax.get_xlim()
            v = np.linspace(v_min, v_max, 1000)
            f = TemperaturePlotter.boltzmann_dist(v, value, temp)
            ax.plot(v, f, **plot_opts)
            ax.legend()

    @staticmethod
    def boltzmann_dist(v, mass, T):
        return 4 * np.pi * (mass / (2 * np.pi * CONSTANTS.K * T)) ** (3 / 2) * v ** 2 * np.exp(-mass * v ** 2 / (2 * CONSTANTS.K * T))


class PlotRightAfterNVE:
    @staticmethod
    def run(path_dump, path_thermo, idx_to_select):
        PlotRightAfterNVE.run_start_of_nvt(path_dump, path_thermo)
        PlotRightAfterNVE.run_end_of_nvt(path_dump, idx_to_select)

    @staticmethod
    def run_start_of_nvt(path_dump, path_thermo):
        idx_to_select, temp = PlotRightAfterNVE.select_idx_and_temp(path_thermo)
        data, _ = DumpReader.read_data(path_dump, idx_to_select)
        data = data[0]
        TemperaturePlotter.plot(data, temp, f'image_{idx_to_select}')

    @staticmethod
    def run_end_of_nvt(path_dump, idx_to_select):
        data, _ = DumpReader.read_data(path_dump, idx_to_select)
        data = data[0]
        temp = PlotRightAfterNVE.check_whole_temperature(data)
        TemperaturePlotter.plot(data, temp, f'image_{idx_to_select}')

    @staticmethod
    def select_idx_and_temp(path_thermo):
        COL_IDX_STEP = 0
        COL_IDX_TIME = 1
        COL_IDX_TEMP = 2
        data = np.loadtxt(path_thermo, skiprows=1, usecols=(COL_IDX_STEP,
                                                            COL_IDX_TIME,
                                                            COL_IDX_TEMP))
        # by checking the first column, remove duplicated values
        # then, select the first value whose second column is larger than 2.0
        # return the index
        NVE_TIME = 2.0
        _, indices = np.unique(data[:, COL_IDX_STEP], return_index=True)
        data = data[indices]
        np.savetxt('thermo_reduced.txt', data)
        idx = np.where(data[:, COL_IDX_TIME] > NVE_TIME)[0][0]
        return idx, data[idx, COL_IDX_TEMP]

    @staticmethod
    def check_whole_temperature(sub_dict):
        kE_sum = 0
        n_dof = 0
        for element, (_, speeds) in sub_dict.items():
            mass = PLOT_OPTS.MASS[element] * CONSTANTS.AMU_TO_KG
            kE = 0.5 * mass * speeds ** 2
            kE_sum += np.sum(kE)
            n_dof += 3 * len(speeds)
        T = 2 * kE_sum / (n_dof * CONSTANTS.K)
        print(f'Temperature: {T} K, atoms: {n_dof / 3}')
        return T


class TempGradientPlotter:
    @staticmethod
    def run_check_temp_gradient(path_dump, path_thermo, idx_to_select, to_plot=False):
        data, dump = DumpReader.read_data(path_dump, None)
        if idx_to_select is not None:
            if idx_to_select < 0:
                idx_to_select = len(data) + idx_to_select
            data, image = data[idx_to_select], dump[idx_to_select]
            T_upper, T_lower = TempGradientPlotter.get_upper_and_lower_temp(
                data, image, idx_to_select, to_plot=to_plot)
        else:
            T_upper_list, T_lower_list = [], []
            for ((dump_idx, sub_dict), image) in zip(data.items(), dump):
                T_upper, T_lower = TempGradientPlotter.get_upper_and_lower_temp(
                    sub_dict, image, dump_idx, to_plot=to_plot)
                T_upper_list.append(T_upper)
                T_lower_list.append(T_lower)
                print(T_upper, T_lower)
            time_list = TempGradientPlotter.get_time_list(path_thermo)
            TempGradientPlotter.plot(T_upper_list, T_lower_list, time_list)

    @staticmethod
    def get_upper_and_lower_temp(data, image, idx_to_select, to_plot=False):
        median = TempGradientPlotter.get_median_height(image, data)
        pos = image.get_positions()
        data_upper, data_lower = {}, {}
        for (symbol, (mask, speeds)) in data.items():
            idx_upper = pos[mask, 2] >= median
            idx_lower = pos[mask, 2] < median
            data_upper[symbol] = (mask[idx_upper], speeds[idx_upper])
            data_lower[symbol] = (mask[idx_lower], speeds[idx_lower])

        T_upper = PlotRightAfterNVE.check_whole_temperature(data_upper)
        T_lower = PlotRightAfterNVE.check_whole_temperature(data_lower)
        if to_plot:
            TemperaturePlotter.plot(data_upper, T_upper, f'image_{idx_to_select}_upper')
            TemperaturePlotter.plot(data_lower, T_lower, f'image_{idx_to_select}_lower')
        return T_upper, T_lower

    @staticmethod
    def get_median_height(image, data):
        idx_total = []
        for (mask, _) in data.values():
            idx_total.append(mask)
        idx_total = np.concatenate(idx_total)
        pos = image.get_positions()
        filtered = pos[idx_total]
        median = np.median(filtered[:, 2])
        return median

    @staticmethod
    def get_time_list(path_thermo):
        COL_IDX_STEP = 0
        COL_IDX_TIME = 1
        COL_IDX_TEMP = 2
        data = np.loadtxt(path_thermo, skiprows=1, usecols=(COL_IDX_STEP,
                                                            COL_IDX_TIME,
                                                            COL_IDX_TEMP))
        # by checking the first column, remove duplicated values
        # then, select the first value whose second column is larger than 2.0
        # return the index
        NVE_TIME = 2.0
        _, indices = np.unique(data[:, COL_IDX_STEP], return_index=True)
        data = data[indices]
        return data[:, COL_IDX_TIME]

    @staticmethod
    def plot(T_upper_list, T_lower_list, time_list):
        fig, ax = plt.subplots()
        ax.plot(time_list, T_upper_list, label='Upper', color='orange')
        ax.plot(time_list, T_lower_list, label='Lower', color='blue')
        ax.set_xlabel('Time [ps]')
        ax.set_ylabel('Temperature [K]')
        ax.legend()

        ax.set_xlim(0, None)
        ax.set_ylim(0, 600)

        ax.axhline(y=300, color='grey', linestyle='--')
        ax.axvline(x=2.0, color='grey', linestyle='--')
        fig.tight_layout()
        fig.savefig('temp_gradient.png')


if __name__ == '__main__':
    path_dump, path_thermo = sys.argv[1], sys.argv[2]
    # PlotRightAfterNVE.run(path_dump, path_thermo, -1)
    TempGradientPlotter.run_check_temp_gradient(path_dump, path_thermo, -1, to_plot=True)
