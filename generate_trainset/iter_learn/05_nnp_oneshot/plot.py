import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse


def plot_one(e_nnp, e_dft, ax, ax_diff, fig_title):
    e_diff = e_nnp - e_dft
    rmse = mse(e_dft, e_nnp, squared=False)
    x = [i for i in range(len(e_diff))]

    ax.scatter(x, e_dft, color='k', s=3)
    ax.plot(x, e_dft, color='k', linestyle='--', label='dft')
    ax.scatter(x, e_nnp, color='r', s=3)
    ax.plot(x, e_nnp, color='r', linestyle='--', label='nnp')
    ax.set_ylabel('pot E (meV/atom)')
    ax.legend(loc='upper left')

    ax.set_title(fig_title)

    ax_diff.scatter(x, e_diff, color='b', s=3)
    ax_diff.plot(x, e_diff, color='b', linestyle='--', label='nnp-dft')
    ax_diff.legend(loc='lower left')
    ax_diff.set_xlabel('step')
    ax_diff.set_ylabel('E diff (meV/atom)')

    idx_list = [i for i in range(len(e_dft))][::5]

    for idx in idx_list:
        ax.axvline(idx, color='grey', linestyle='--', alpha=0.3)
        ax_diff.axvline(idx, color='grey', linestyle='--', alpha=0.3)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textbox = f"Trajectory RMSE : {rmse:.0f} (meV/atom)"
    ax.text(0.95, 0.05, textbox,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes, bbox=props)


def plot_all(e_dft, e_nnp, nions):
    e_dft = np.divide(e_dft, nions) * 1000
    e_nnp = np.divide(e_nnp, nions) * 1000

    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(4, 4, figsize=(30, 12))
    axes = axes.flatten('F')
    ax_list, ax_diff_list = axes[::2], axes[1::2]

    name_list = [
        'CF_10',
        'CF_30',
        'CF3_10',
        'CF3_30',
        'CH2F_10',
        'CH2F_30',
        'CHF2_10',
        'CHF2_30',
    ]

    idx_start = 0
    shift = 250
    idx_end = idx_start+shift

    for ax, ax_diff, fig_title in zip(ax_list, ax_diff_list, name_list):
        e_dft_new = e_dft[idx_start:idx_end]
        e_nnp_new = e_nnp[idx_start:idx_end]
        plot_one(e_nnp_new, e_dft_new, ax, ax_diff, fig_title)

        print(f"idx_start : {idx_start}, idx_end : {idx_end}")
        idx_start += shift
        idx_end += shift

    plt.grid(visible=True, axis='y')
    fig.tight_layout()
    fig.savefig('Traj_RMSE.png')


# def get_idx_iter_start():
#     iters = np.loadtxt("../energy_dft.dat", usecols=[2])
#     iter_current = -1
#     idx_list = []
#     for idx, iter_count in enumerate(iters):
#         if iter_current != iter_count:
#             idx_list.append(idx)
#             iter_current = iter_count

#     return idx_list


def main():
    nions = np.loadtxt("../energy_dft.dat", usecols=[5])
    e_dft = np.loadtxt("../energy_dft.dat", usecols=[4])
    e_nnp = np.loadtxt("energy_nnp.dat")

    plot_all(e_dft, e_nnp, nions)


main()
