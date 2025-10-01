import numpy as np
import matplotlib.pyplot as plt


def main():
    time_list = []
    n_Si_list = []
    n_O_list = []
    n_C_list = []
    n_H_list = []
    n_F_list = []

    for i in range(1, 51):
        mat = np.loadtxt(
            f'thermo_{i}.dat', skiprows=2, usecols=(1, 6, 7, 8, 9, 10))

        time_list.append(mat[:, 0])
        n_Si_list.append(mat[:, 1])
        n_O_list.append(mat[:, 2])
        n_C_list.append(mat[:, 3])
        n_H_list.append(mat[:, 4])
        n_F_list.append(mat[:, 5])

    time_tot = np.concatenate(time_list)
    n_Si_tot = np.concatenate(n_Si_list)
    n_O_tot = np.concatenate(n_O_list)
    n_C_tot = np.concatenate(n_C_list)
    n_H_tot = np.concatenate(n_H_list)
    n_F_tot = np.concatenate(n_F_list)

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_tot, n_Si_tot, label='Si')
    ax.plot(time_tot, n_O_tot, label='O')
    ax.plot(time_tot, n_C_tot, label='C')
    ax.plot(time_tot, n_H_tot, label='H')
    ax.plot(time_tot, n_F_tot, label='F')
    ax.set_xlabel('time (ps)')
    ax.set_ylabel('number of atoms')
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('number_of_atoms.png')


main()
