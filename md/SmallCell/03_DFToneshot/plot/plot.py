import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def read_energy_dft(folder):
    energy = None
    with open(folder, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "free  " in line:
            energy = float(line.split()[-2])
            break
    return energy


def read_energy_gnn(folder):
    energy = None
    with open(folder, "r") as f:
        lines = f.readlines()
        energy = float(lines[-1].split()[1])
    return energy


def read_nions_dft(folder):
    nions = None
    with open(folder, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "NIONS" in line:
            nions = int(line.split()[-1])
            break
    return nions


def read_data(src_dft, cal_type=None):
    if os.path.isfile(f"data_{cal_type}.pkl"):
        with open(f"data_{cal_type}.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    ion_ionE_list = [i for i in os.listdir(src_dft) if
              os.path.isdir(os.path.join(src_dft, i))]

    if cal_type == "DFT":
        file_name = "OUTCAR"
        read_energy = read_energy_dft
    elif cal_type == "GNN":
        file_name = "thermo.dat"
        read_energy = read_energy_gnn
    else:
        raise ValueError("cal_type should be DFT or GNN")

    data = {}
    for ion_ionE in ion_ionE_list:
        if data.get(ion_ionE) is None:
            data[ion_ionE] = []

        file_paths = [
            os.path.join(src_dft, ion_ionE, i, file_name)
            for i in os.listdir(os.path.join(src_dft, ion_ionE)) if
            os.path.isdir(os.path.join(src_dft, ion_ionE, i))]
        file_paths.sort()
        print(cal_type, ion_ionE, len(file_paths))

        for file_path in file_paths:
            energy = read_energy(file_path)
            if cal_type == "DFT":
                nions = read_nions_dft(file_path)
                energy = (energy, nions)
            data[ion_ionE].append(energy)
            print(f"{file_path} : {energy} eV")

    with open(f"data_{cal_type}.pkl", "wb") as f:
        pickle.dump(data, f)

    return data


def plot(data_dft, data_gnn):
    props_dft = {
        "label": "DFT",
        "color": "black",
        "linestyle": "-",
        "marker": "o",
        "alpha": 0.5,
        }
    props_gnn = {
        "label": "GNN",
        "color": "blue",
        "linestyle": "--",
        "marker": "x",
        "alpha": 0.5,
        }
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for (ion_ionE, E_dft_nion), ax in zip(data_dft.items(), axes):
        E_dft = np.array([i[0] for i in E_dft_nion])
        nions = np.array([i[1] for i in E_dft_nion])
        E_gnn = np.array(data_gnn[ion_ionE])
        E_dft = E_dft / nions
        E_gnn = E_gnn / nions
        x = np.arange(len(E_dft))

        ax.plot(x, E_dft, **props_dft)
        ax.plot(x, E_gnn, **props_gnn)

        ax.set_title(ion_ionE)
        ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95))
        ax.set_xlabel("selected step")
        ax.set_ylabel("energy (eV/atom)")

    y_min = min([i.get_ylim()[0] for i in axes])
    y_max = max([i.get_ylim()[1] for i in axes])
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig.savefig("result.png")


def plot_diff(data_dft, data_gnn):
    props = {
        "color": "orange",
        "linestyle": "-",
        "marker": "o",
        "alpha": 0.5,
        }
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for (ion_ionE, E_dft_nion), ax in zip(data_dft.items(), axes):
        nions = np.array([i[1] for i in E_dft_nion])
        E_dft = np.array([i[0] for i in E_dft_nion])
        E_gnn = np.array(data_gnn[ion_ionE])
        E_dft = E_dft / nions
        E_gnn = E_gnn / nions
        x = np.arange(len(E_dft))

        diff = E_dft - E_gnn
        diff *= 1000
        ax.plot(x, diff, **props)

        ax.set_title(ion_ionE)
        ax.set_xlabel("selected step")
        ax.set_ylabel("energy diff (meV/atom)")

    y_max = max([i.get_ylim()[1] for i in axes])
    for ax in axes:
        ax.set_ylim(0, y_max)

    fig.tight_layout()
    fig.savefig("result_diff.png")


def main():
    src_dft = "/data2/andynn/ZBL_modify/SmallCell/03_DFToneshot/results/dft/"
    src_gnn = "/data2/andynn/ZBL_modify/SmallCell/03_DFToneshot/results/gnn/"

    data_dft = read_data(src_dft, cal_type="DFT")
    data_gnn = read_data(src_gnn, cal_type="GNN")

    plt.rcParams.update({'font.size': 18})
    plot(data_dft, data_gnn)
    plot_diff(data_dft, data_gnn)


if __name__ == "__main__":
    main()
