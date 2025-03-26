import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from ase.io import read
import pickle

class EtchYieldCalculator(ABC):
    def __init__(self, src, n_traj, dst, interval=10):
        self.src = src
        self.n_traj = n_traj
        self.interval = interval
        self.dst = dst

    def run(self):
        src = f"{self.dst}.dat"
        n_Si_etched = None
        if os.path.exists(src):
            n_Si_etched = np.loadtxt(src)
        else:
            n_Si_etched = self._get_deleted_Si()
            np.savetxt(src, n_Si_etched)
        etch_yield = self._get_interval_average(n_Si_etched)
        norm_factor = self._get_normalize_factor()

        return norm_factor, n_Si_etched, etch_yield

    def _get_normalize_factor(self):
        path_input_structure = next(p for p in
            [f"{self.src}/str_shoot_0_after_mod.coo",
             f"{self.src}/str_shoot_0.coo"]
            if os.path.isfile(p))
        with open(path_input_structure, "r") as f:
            lines = f.readlines()
            lat_x, lat_y = map(int, [lines[5].split()[1], lines[6].split()[1]])
        return 1 / (lat_x * lat_y)

    def _get_interval_average(self, x):
        x_new = [0]
        for idx_end in range(1, len(x)):
            idx_start = max(0, idx_end - self.interval)
            dat = (x[idx_end] - x[idx_start]) / (idx_end - idx_start)
            x_new.append(dat)
        return np.array(x_new)

    @abstractmethod
    def _get_deleted_Si(self):
        return

class EtchYieldPlotter(ABC):
    def __init__(self, dst, norm_factor, n_Si_etched, etch_yield):
        self.dst = dst
        self.norm_factor = norm_factor
        self.n_Si_etched = n_Si_etched
        self.etch_yield = etch_yield

    def run(self):
        plt.rcParams.update({'font.size': 16})
        fig, (ax_Si, ax_yield) = plt.subplots(2, 1, figsize=(8, 6))

        x = np.arange(len(self.n_Si_etched)) * self.norm_factor
        ax_Si.plot(x, self.n_Si_etched, color='black')
        ax_Si.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
        ax_Si.set_ylabel("number of etched Si")

        x_yield = [i for i in range(len(self.etch_yield)) if self.etch_yield[i] >= 0]
        x_yield = np.array(x_yield) * self.norm_factor
        y_yield = [y for y in self.etch_yield if y >= 0]
        ax_yield.plot(x_yield, y_yield, color='orange')
        ax_yield.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
        ax_yield.set_ylabel("Etch yield (Si/ion)")

        # yield_avg = np.mean(y_yield[-self.interval:])
        yield_avg = y_yield[-1]
        textbox = f"yield = {yield_avg:.3f}"
        ax_yield.text(0.95, 0.05, textbox,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      transform=ax_yield.transAxes)

        fig.tight_layout()
        fig.savefig(f'{self.dst}.png')

        with open(f'{self.dst}_plot_data.pkl', 'wb') as f:
            plot_data = {
                    'x': x,
                    'n_Si_etched': self.n_Si_etched,
                    'x_yield': x_yield,
                    'y_yield': y_yield
                    }
            pickle.dump(plot_data, f)

class EYCalculatorFromStructure(EtchYieldCalculator):
    def _get_deleted_Si(self):
        n_Si_etched = [0]
        for i in range(1, self.n_traj + 1):
            path_structure_now = f"{self.src}/rm_byproduct_str_shoot_{i}.coo"
            path_structure_prev = f"{self.src}/str_shoot_{i-1}_after_mod.coo"
            n_Si_now = self._get_Si_count(path_structure_now)
            n_Si_prev = self._get_Si_count(path_structure_prev)

            n_diff = n_Si_prev - n_Si_now if n_Si_prev > n_Si_now else 0
            n_Si_etched.append(n_diff)
            print(f'{path_structure_now} Complete, {n_diff}')

        y = np.cumsum(n_Si_etched)

        return np.array(y)

    @staticmethod
    def _get_Si_count(path_structure):
        ATOM_IDX_1 = 1
        ATOM_IDX_2 = 2
        ATOM_IDX_3 = 3
        ATOM_IDX_4 = 4
        ATOM_IDX_5 = 5

        ELEM_IDX_Si = 14
        ELEM_IDX_O = 8
        ELEM_IDX_C = 6
        ELEM_IDX_H = 1
        ELEM_IDX_F = 9

        lmp_dat_opts= {
                'format': 'lammps-data',
                'Z_of_type': {
                    ATOM_IDX_1: ELEM_IDX_Si,
                    ATOM_IDX_2: ELEM_IDX_O,
                    ATOM_IDX_3: ELEM_IDX_C,
                    ATOM_IDX_4: ELEM_IDX_H,
                    ATOM_IDX_5: ELEM_IDX_F,
                    },
                'atom_style': 'atomic'
                }
        atoms = read(path_structure, **lmp_dat_opts)
        return len([atom for atom in atoms if atom.symbol == 'Si'])

class EYCalculatorFromStructureDistributed(EYCalculatorFromStructure):
    def __init__(self, src_list, dst, interval=10):
        self.src_list = src_list
        self.interval = interval
        self.dst = dst

    def _gen_path_dict(self):
        path_dict_now = {}
        path_dict_prev = {}
        for folder in self.src_list:
            for file in os.listdir(folder):
                if not file.endswith(".coo"):
                    continue
                if "_before_anneal" in file:
                    continue

                if "rm_byproduct" in file:
                    idx = int(file.replace('rm_byproduct_str_shoot_', '').replace('.coo', ''))
                    path_dict_now[idx] = f"{folder}/{file}"
                elif "after_mod" in file:
                    idx = int(file.replace('str_shoot_', '').replace('_after_mod.coo', ''))
                    path_dict_prev[idx] = f"{folder}/{file}"
        n_traj = max(path_dict_now.keys())
        return path_dict_now, path_dict_prev, n_traj

    def _get_deleted_Si(self):
        n_Si_etched = [0]
        path_dict_now, path_dict_prev, n_traj = self._gen_path_dict()
        for i in range(1, n_traj + 1):
            path_structure_now = path_dict_now[i]
            path_structure_prev = path_dict_prev[i-1]
            n_Si_now = self._get_Si_count(path_structure_now)
            n_Si_prev = self._get_Si_count(path_structure_prev)

            n_diff = n_Si_prev - n_Si_now if n_Si_prev > n_Si_now else 0
            n_Si_etched.append(n_diff)
            print(f'{path_structure_now} Complete, {n_diff}')

        y = np.cumsum(n_Si_etched)

        return np.array(y)

    def _get_normalize_factor(self):
        src = self.src_list[0]
        path_input_structure = next(p for p in
            [f"{src}/str_shoot_0_after_mod.coo",
             f"{src}/str_shoot_0.coo"]
            if os.path.isfile(p))
        with open(path_input_structure, "r") as f:
            lines = f.readlines()
            lat_x, lat_y = map(int, [lines[5].split()[1], lines[6].split()[1]])
        return 1 / (lat_x * lat_y)


# class EYCalculatorVer1(EtchYieldCalculator):
#     def run(self):
#         src = f"{self.dst}.dat"
#         n_Si_etched = None
#         if os.path.exists(src):
#             n_Si_etched = np.loadtxt(src)
#         else:
#             n_Si_etched = self._get_deleted_Si()
#             n_Si_etched = self._get_deleted_Si_by_molecules(n_Si_etched)
#             np.savetxt(src, n_Si_etched)
#         etch_yield = self._get_interval_average(n_Si_etched)
#         norm_factor = self._get_normalize_factor()

#         return norm_factor, n_Si_etched, etch_yield

#     def _get_deleted_Si_by_molecules(self, n_Si_etched):
#         src = f"{self.src}/delete.log"
#         if not os.path.isfile(src):
#             return

#         print("Reading delete.log...")
#         deleted_Si_list = []
#         with open(src, "r") as f:
#             for line in f:
#                 parts = line.split()
#                 idx_iter, Si_count = int(parts[1]), int(parts[4].strip(','))
#                 if Si_count:
#                     deleted_Si_list.append([idx_iter, Si_count])
#         deleted_Si_list = np.array(deleted_Si_list)
#         for i, j in deleted_Si_list:
#             n_Si_etched[i:] += j
#         return n_Si_etched

# class EYCalculatorFromThermo(EYCalculatorVer1):
#     def _get_deleted_Si(self):
#         n_Si_etched_total = 0
#         y = [0]
#         for i in range(1, self.n_traj + 1):
#             mat = np.loadtxt(f'{self.src}/thermo_{i}.dat', skiprows=2, usecols=(6,))
#             n_Si_etched = mat[0] - mat[-1]
#             n_Si_etched_total += n_Si_etched
#             y.append(n_Si_etched_total)
#             print(f'thermo_{i}.dat Complete')
#         return np.array(y)

# class EYCalculatorFromDump(EYCalculatorVer1):
#     def _get_deleted_Si(self):
#         path_desorbed = f"desorption_graph.dat"
#         if not os.path.exists(path_desorbed):
#             print("desorption_graph.dat not found. Returning zero array.")
#             y = [0] * (self.n_traj + 1)
#             return np.array(y)

#         n_Si_etched_dict = {k: 0 for k in range(self.n_traj + 1)}

#         with open(path_desorbed, "r") as f:
#             for line in f:
#                 if line.startswith("--"):
#                     continue
#                 incidence, composition, *_ = line.split('/')
#                 incidence = int(incidence)
#                 composition = [int(i) for i in composition.split()]
#                 n_Si_etched = composition[0]  # Si is the first element in the composition
#                 if n_Si_etched:
#                     n_Si_etched_dict[incidence] += n_Si_etched

#         y = [0]
#         for i in range(1, self.n_traj + 1):
#             y.append(y[-1] + n_Si_etched_dict[i])
#             print(f"dump_{i}.dat Complete")

#         return np.array(y)

# def run_EYplotterFromThermoDat():
#     src = sys.argv[1]
#     n_traj = int(sys.argv[2])
#     interval = int(sys.argv[3])
#     dst = sys.argv[4]
#     plotter1 = EYPlotterFromThermoDat(src, n_traj, dst, interval=interval)
#     plotter1.run()

# def run_EYplotterFromDump():
#     src = sys.argv[1]
#     n_traj = int(sys.argv[2])
#     interval = int(sys.argv[3])
#     dst = sys.argv[4]
#     plotter2 = EYPlotterFromDump(src, n_traj, interval=interval)
#     plotter2.run()

# def run():
#     if len(sys.argv) != 5:
#         print("Usage: python etchyield.py [src] [n_traj] [interval] [dst]")
#         sys.exit(1)
#     src = sys.argv[1]
#     n_traj = int(sys.argv[2])
#     interval = int(sys.argv[3])
#     dst = sys.argv[4]

#     calculator = EYCalculatorFromStructure(src, n_traj, dst, interval=interval)
#     norm_factor, n_Si_etched, etch_yield = calculator.run()
#     plotter = EtchYieldPlotter(dst, norm_factor, n_Si_etched, etch_yield)
#     plotter.run()


def run_distributed():
    if len(sys.argv) < 4:
        print("Usage: python etchyield.py [interval] [dst] [src1] [src2] ...")
        sys.exit(1)

    interval = int(sys.argv[1])
    dst = sys.argv[2]
    src_list = sys.argv[3:]

    calculator = EYCalculatorFromStructureDistributed(src_list, dst, interval=interval)
    norm_factor, n_Si_etched, etch_yield = calculator.run()
    plotter = EtchYieldPlotter(dst, norm_factor, n_Si_etched, etch_yield)
    plotter.run()


if __name__ == "__main__":
    run_distributed()
