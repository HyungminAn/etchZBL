import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from ase.io import read


def read_thermo_dat(path_file):
    with open(path_file, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            if line[0] == '#':
                continue
            pe = float(line.split()[2])
            data.append(pe)

    return data


def read_dump_force(path_dump):
    atoms = read(path_dump, index=':')
    forces = []
    for atom in atoms:
        forces.append(atom.get_forces())
    return forces


def read_data(src, data_to_read, label):
    print(f"Reading data from {src}...", end='')
    if os.path.exists(f"data_{label}.pkl"):
        with open(f"data_{label}.pkl", "rb") as f:
            data = pickle.load(f)
        print("Done.")
        return data

    dump_list = [f"{src}/dump_{i}.lammps" for i in data_to_read]
    thermo_list = [f"{src}/thermo_{i}.dat" for i in data_to_read]

    energy, force = [], []
    for dump, thermo in zip(dump_list, thermo_list):
        if not os.path.exists(dump) or not os.path.exists(thermo):
            print(f"Error: {dump} or {thermo} does not exist.")
            sys.exit(1)
        energy.append(read_thermo_dat(thermo))
        force.append(read_dump_force(dump))

    data = [energy, force]
    print("Done.")
    with open(f"data_{label}.pkl", "wb") as f:
        pickle.dump(data, f)
    return data


def process_data(data_1, data_2):
    print("Processing data...", end='')
    energy_1, force_1 = data_1
    energy_2, force_2 = data_2

    mark = [0] + [len(i) for i in energy_1]
    mark = np.cumsum(mark)

    energy_1 = np.concatenate(energy_1)
    energy_2 = np.concatenate(energy_2)
    energy_diff = energy_1 - energy_2

    force_1 = [f for f_list in force_1 for f in f_list]
    force_2 = [f for f_list in force_2 for f in f_list]
    force_diff = [np.max(np.abs(f1 - f2)) for f1, f2 in zip(force_1, force_2)]

    print("Done.")
    return energy_1, energy_2, energy_diff, force_diff, mark


def plot_mark(ax, mark):
    for m in mark:
        ax.axvline(x=m, color='g', linestyle='--', alpha=0.5)


def plot(data1, data2, label1, label2):
    print("Plotting...", end='')
    energy_1, energy_2, energy_diff, force_diff, mark = process_data(data1, data2)

    fig, (ax, ax_e_diff, ax_f_diff) = plt.subplots(3, 1, figsize=(8, 12))
    ax.plot(energy_1, label=label1, color='b', marker='o')
    ax.plot(energy_2, label=label2, color='r', marker='x')
    ax.legend(loc='upper left')
    ax.set_ylabel('Potential Energy (eV)')
    ax.set_xlabel('Step')

    ax_e_diff.plot(energy_diff, color='g', linestyle='--', marker='o')
    ax_e_diff.set_ylabel('Energy difference (eV)')
    ax_e_diff.set_xlabel('Step')

    ax_f_diff.plot(force_diff, color='m', linestyle='--', marker='o')
    ax_f_diff.set_ylabel('Max force difference (eV/Angstrom)')
    ax_f_diff.set_xlabel('Step')

    plot_mark(ax, mark)
    plot_mark(ax_e_diff, mark)
    plot_mark(ax_f_diff, mark)

    fig.tight_layout()
    fig.savefig('result.png')

    print("Done.")


def main():
    if len(sys.argv) != 5:
        print('Usage: python compare.py <src_1> <src_2> <label_1> <label_2>')
        sys.exit(1)

    data_to_read = ['premelt', 'melt', 'quench', 'anneal']
    src1, src2, label1, label2 = sys.argv[1:5]
    data1 = read_data(src1, data_to_read, label1)
    data2 = read_data(src2, data_to_read, label2)
    plot(data1, data2, label1, label2)


if __name__ == '__main__':
    main()
