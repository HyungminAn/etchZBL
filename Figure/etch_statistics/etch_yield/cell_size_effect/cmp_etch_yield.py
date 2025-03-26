import numpy as np
import matplotlib.pyplot as plt


def get_deleted_Si_by_molecules(target_folder):
    path_delete_log = f"{target_folder}/delete.log"
    with open(path_delete_log, "r") as f:
        lines = f.readlines()

    deleted_Si_list = []
    for line in lines:
        line = line.split()
        idx_iter = int(line[1])
        Si_count = int(line[4].strip(','))

        if Si_count:
            deleted_Si_list.append([idx_iter, Si_count])

    deleted_Si_list = np.array(deleted_Si_list)

    return deleted_Si_list


def get_deleted_Si_during_MD(target_folder, n_traj):
    n_Si_etched_total = 0
    y = [0]

    for i in range(1, n_traj+1):
        mat = np.loadtxt(
            f'{target_folder}/thermo_{i}.dat', skiprows=2, usecols=(6, ))
        n_Si_init, n_Si_final = mat[0], mat[-1]
        n_Si_etched = n_Si_init - n_Si_final
        n_Si_etched_total += n_Si_etched

        y.append(n_Si_etched_total)
        print(f'thermo_{i}.dat Complete')

    y = np.array(y)

    return y


def get_interval_average(x, interval):
    x_new = []
    for idx_end in range(len(x)):
        idx_start = idx_end - interval

        cond = idx_start < 0
        if cond:
            idx_start = 0

        if idx_end == 0:
            x_new.append(0)
            continue

        dat = (x[idx_end] - x[idx_start]) / (idx_end - idx_start)
        x_new.append(dat)

    return np.array(x_new)


def get_etch_yield(target_folder, n_traj):
    interval = 400
    n_Si_etched = get_deleted_Si_during_MD(target_folder, n_traj)
    deleted_Si_by_molecules = get_deleted_Si_by_molecules(target_folder)

    for i, j in deleted_Si_by_molecules:
        n_Si_etched[i:] += j

    # Get etch yield
    etch_yield = get_interval_average(n_Si_etched, interval)

    return n_Si_etched, etch_yield


def get_normalize_factor(target_folder):
    path_input_structure = f"{target_folder}/CHF_shoot_0_after_removing.coo"
    with open(path_input_structure, "r") as f:
        lines = f.readlines()

    lat_x = int(lines[5].split()[1])  # in Angstrom unit
    lat_y = int(lines[6].split()[1])  # in Angstrom unit

    area = lat_x * lat_y  # in A^2 unit
    return 1 / area


def get_plot_figure():
    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    ax_Si, ax_yield = axes
    ax_Si.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
    ax_Si.set_ylabel(r"etched ion ($ \times 10^{16} \mathrm{cm}^{-2}$)")
    ax_yield.set_xlabel(r"ion dose ($ \times 10^{16} \mathrm{cm}^{-2}$)")
    ax_yield.set_ylabel("Etch yield (Si/ion)")
    return fig, axes


def plot(
        n_Si_etched, etch_yield, norm_factor,
        fig, axes, target_folder, plot_color):
    ax_Si, ax_yield = axes
    label = ' '.join(target_folder.split('/')[-2:])

    x = np.array([i for i in range(len(n_Si_etched))], dtype=float)
    x *= norm_factor
    n_Si_etched *= norm_factor
    ax_Si.plot(x, n_Si_etched, color=plot_color, label=label)

    x = [i for i in range(len(etch_yield)) if etch_yield[i] >= 0]
    etch_yield = [x for x in etch_yield if x >= 0]
    x = np.array(x, dtype=float) * norm_factor
    ax_yield.plot(x, etch_yield, color=plot_color, label=label)


def main():
    n_traj = 800
    fig, axes = get_plot_figure()
    folder_list = [
        '../../02_Run/100eV',
        '../../03_RunLargeCell_334/100eV',
        '../../04_RunLargeCell_445/100eV'
    ]
    color_list = ['red', 'blue', 'grey']

    for target_folder, plot_color in zip(folder_list, color_list):
        if target_folder.split('/')[-2].endswith('334'):
            n_Si_etched, etch_yield = get_etch_yield(target_folder, 1800)
        elif target_folder.split('/')[-2].endswith('445'):
            n_Si_etched, etch_yield = get_etch_yield(target_folder, 2000)
        else:
            n_Si_etched, etch_yield = get_etch_yield(target_folder, n_traj)

        norm_factor = get_normalize_factor(target_folder)
        plot(
            n_Si_etched, etch_yield, norm_factor,
            fig, axes, target_folder, plot_color,
            )

    ax_Si, ax_yield = axes
    ax_Si.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig('etch_yield_cmp.png')


main()
