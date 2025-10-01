import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class TYPES:
    IONS = ['CF', 'CF3', 'CH2F', 'CHF2']
    ENERGIES = [10, 30]
    ITERS = [1, 2, 3]
    INCIDENCES = [i for i in range(1, 51)]


def get_data():
    result = {}
    for iter in TYPES.ITERS:
        for ion in TYPES.IONS:
            for energy in TYPES.ENERGIES:
                for incidence in TYPES.INCIDENCES:
                    path = f'src/Iter_{iter}/02_NNP_RIE/{ion}/{energy}/thermo_{incidence}.dat'
                    if os.path.exists(path):
                        print(f"File exists: {path}")
                    dat = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2))
                    dat[:, 0] -= dat[0, 0]  # Normalize step
                    dat[:, 1] -= dat[0, 1]  # Normalize time
                    key = (iter, ion, energy, incidence)
                    result[key] = dat
                    print(f"Loaded data from {path}")
    return result


def plot(data):
    plt.rcParams.update({'font.family': 'arial', 'font.size': 14})
    n_ions = len(TYPES.IONS)
    n_energies = len(TYPES.ENERGIES)
    fig, axes = plt.subplots(n_energies, n_ions, figsize=(5 * n_ions, 5 * n_energies))
    ax_dict = {}
    for ion in TYPES.IONS:
        for energy in TYPES.ENERGIES:
            ax_dict[(energy, ion)] = axes[TYPES.ENERGIES.index(energy), TYPES.IONS.index(ion)]

    for ion in TYPES.IONS:
        for energy in TYPES.ENERGIES:
            for incidence in TYPES.INCIDENCES:
                key = (1, ion, energy, incidence)
                if key not in data:
                    continue
                dat = data[key]
                ax = ax_dict[(energy, ion)]

                step, time, temp = dat[:, 0], dat[:, 1], dat[:, 2]
                ax.plot(time, temp, color='orange', alpha=0.1)
                print(f"Plotting data for {ion} {energy} eV incidence {incidence}")

    for ion in TYPES.IONS:
        for energy in TYPES.ENERGIES:
            for incidence in TYPES.INCIDENCES:
                key = (1, ion, energy, incidence)
                if key not in data:
                    continue
                dat = data[key]
                ax = ax_dict[(energy, ion)]

                step, time, temp = dat[:, 0], dat[:, 1], dat[:, 2]
                ax.scatter(time[-1], temp[-1], marker='^',
                        edgecolors='black', facecolors='orange', s=40)

    for (energy, ion), ax in ax_dict.items():
        ax.set_title(f'{ion} {energy} eV')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Temperature (K)')
        ax.axhline(350, color='grey', linestyle='--', label='350 K')
        ax.axhline(300, color='red', linestyle='--', label='300 K')
        ax.set_xlim(2.0, 4.5)
        ax.set_ylim(200, 400)

    fig.tight_layout()
    fig.savefig('result.png')


def main():
    data = get_data()
    plot(data)


if __name__ == "__main__":
    main()
