import os
import matplotlib.pyplot as plt
import numpy as np


def read_data_NNP(folder_list):
    energies = []
    for folder in folder_list:
        with open(os.path.join(folder, "log.lammps"), "r") as f:
            lines = f.readlines()

        for line in lines[::-1]:
            if line.startswith("my_pe"):
                energy = float(line.split()[-2])
                energies.append(energy)
                print(f"Energy_NNP: {energy}")
                break

    return np.array(energies)


def read_data_DFT(folder_list):
    energies = []
    for folder in folder_list:
        path = os.path.join(folder, "DFT", "OUTCAR")
        if not os.path.exists(path):
            print(f"OUTCAR not found in {folder}")
            continue

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "free  " in line:
                energy = float(line.split()[-2])
                energies.append(energy)
                print(f"Energy_DFT: {energy}")
                break

    return np.array(energies)


def plot(E_dict):
    x = E_dict['bond_length']
    fig, ax = plt.subplots()

    nnp_props = {
            'color': 'blue',
            'marker': 'o',
            'label': 'NNP',
            'linestyle': '--',
            }
    dft_props = {
            'color': 'red',
            'marker': 'x',
            'label': 'DFT',
            'linestyle': '--',
            }

    y_nnp = E_dict['E_NNP']
    y_nnp -= y_nnp[0]
    ax.plot(x, y_nnp, **nnp_props)
    ax.text(x[-1], y_nnp[-1], f"{y_nnp[-1]:.2f} eV", fontsize=12)

    y_dft = E_dict['E_DFT']
    y_dft -= y_dft[0]
    ax.plot(x[:len(y_dft)], y_dft, **dft_props)
    ax.text(x[-1], y_dft[-1], f"{y_dft[-1]:.2f} eV", fontsize=12)

    ax.set_xlabel("Shift from Equilibrium Bond Length ($\AA$)")
    ax.set_ylabel("Relative energy (eV)")
    ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05))

    fig.tight_layout()
    fig.savefig("result.png")


def main():
    src = "../CF"
    folder_list = sorted([os.path.join(src, folder)
                         for folder in os.listdir(src)
                         if os.path.isdir(os.path.join(src, folder))],
                         key=lambda x: float(x.split("_")[-1]))
    bond_length = [float(folder.split("_")[-1]) for folder in folder_list]
    E_dict = {
            "E_NNP": read_data_NNP(folder_list),
            "E_DFT": read_data_DFT(folder_list),
            "bond_length": bond_length,
            }
    plot(E_dict)


if __name__ == "__main__":
    main()
