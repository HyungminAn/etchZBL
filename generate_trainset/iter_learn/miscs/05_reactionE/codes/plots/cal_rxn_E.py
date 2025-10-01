import os
import sys
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class EnergyNotCalculatedError(Exception):
    pass


def get_energy(
        rxn_info, src, path_gas, cal_type,
        ):

    if ',' in rxn_info:
        idx, gas_list = rxn_info.split(",")
    else:
        idx, gas_list = rxn_info[0], []

    incidence = idx.split("_")[0]
    path_slab_E = f"{src}/post_process_bulk_gas/{incidence}/{idx}/{cal_type}/e"
    if not os.path.exists(path_slab_E):
        raise EnergyNotCalculatedError(f"Energy of {idx} not calculated")
    with open(path_slab_E, 'r') as f:
        line = f.readline()
        E_slab = float(line.split()[-1])
    E_gas = 0

    if not gas_list:
        E_rxn = E_slab + E_gas
        return E_rxn

    for gas_type in gas_list:
        idx = gas_type.split("_")
        path_gas_E = f"{path_gas}/{idx}/{cal_type}/e"
        if not os.path.exists(path_gas_E):
            raise EnergyNotCalculatedError(f"Energy of {gas_type} not calculated")
        with open(path_gas_E, 'r') as f:
            E_gas += float(f.readline().split()[-1])

    E_rxn = E_slab + E_gas
    return E_rxn


def get_data(src, cal_type_1, cal_type_2):
    '''
    example of `rxn.dat`:
        #reactants,/products,/add_to_rxn_E
        1_25/1_151,CO
        1_151,CF/2_25
        ...

    example of `rxn_list`:
        [
            ['1_25', '1_151,CO'],
            ['1_151,CF', '2_25'],
            ...
        ]
    '''
    path_save = f"{src}/rxn_E_{cal_type_1}_{cal_type_2}.pkl"
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {path_save}")
        return data

    path_rxn_E = f"{src}/rxn.dat"
    with open(path_rxn_E,'r') as f:
        # rxn_list = [line.split()[0].split("/") for line in f.readlines()[1:]]
        rxn_list = [line.split()[0].split("/") for line in f.readlines()[1:] if
                    'needs' not in line]

    path_gas = f"{src}/post_process_bulk_gas/gas"
    n_rxn = len(rxn_list)

    E_1, E_2 = np.zeros(n_rxn), np.zeros(n_rxn)
    for i in range(n_rxn):
        rxn_info_before, rxn_info_after = rxn_list[i]
        rxn_info_before = rxn_info_before.split(",")
        rxn_info_after = rxn_info_after.split(",")

        try:
            E_before_1 = get_energy(
                rxn_info_before, src, path_gas, cal_type_1,
                )
            E_before_2 = get_energy(
                rxn_info_before, src, path_gas, cal_type_2,
                )
            E_after_1 = get_energy(
                rxn_info_after, src, path_gas, cal_type_1,
                )
            E_after_2 = get_energy(
                rxn_info_after, src, path_gas, cal_type_2,
                )
        except EnergyNotCalculatedError as e:
            print(f"Error: {e}")
            continue

        E_1[i] = E_after_1 - E_before_1
        E_2[i] = E_after_2 - E_before_2

    data = [E_1, E_2]
    with open(path_save, 'wb') as f:
        pickle.dump(data, f)
        print(f"Data saved to {path_save}")
    return data


def sort_data(data):
    E_1, E_2 = data
    E_1, E_2 = np.array(E_1), np.array(E_2)
    values = np.vstack((E_1, E_2))
    kernel = gaussian_kde(values)
    kernel.set_bandwidth(bw_method=kernel.factor*3)
    Z = kernel(values)
    Z /= np.min(Z[np.nonzero(Z)])
    idx = Z.argsort()
    DFT_rxn, NNP_rxn, Z = E_1[idx], E_2[idx], Z[idx]

    return DFT_rxn, NNP_rxn, Z


def plot(data, cal_type_1, cal_type_2, fig_title):
    print("plotting...", end='')
    fig, ax = plt.subplots()
    fig.set_size_inches(2.5,2)
    ax.set_aspect('equal','box')
    rxn_1, rxn_2, Z = sort_data(data)

    cmap = plt.get_cmap('Blues')
    norm = matplotlib.colors.LogNorm(vmin=1)
    plot_options = {
        'c': Z+1,
        'cmap': cmap,
        's': 1,
        'norm': norm,
        'zorder': 2,
        }
    ax.scatter(rxn_1, rxn_2, **plot_options)
    x_min = np.min([np.min(rxn_1), np.min(rxn_2)])
    x_max = np.max([np.max(rxn_1), np.max(rxn_2)])
    if x_min < 0:
        x_min *= 1.1
    else:
        x_min *= 0.9
    if x_max < 0:
        x_max *= 0.9
    else:
        x_max *= 1.1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.axline((0, 0), slope=1, linestyle='--', zorder=1, color='grey')
    ax.set_aspect('equal')
    # ax.plot(box_range, box_range, 'k--', zorder=1)
    cal_type_1 = cal_type_1.upper()
    cal_type_2 = cal_type_2.upper()
    x_label = fr"$E^{{\rm{{{cal_type_1}}}}}_{{\rm{{rxn}}}}$ (eV)"
    y_label = fr"$E^{{\rm{{{cal_type_2}}}}}_{{\rm{{rxn}}}}$ (eV)"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    E_1, E_2 = data
    mae = np.mean(np.abs(E_1-E_2))
    pearson = np.corrcoef(E_1,E_2)[0,1]
    r2 = 1-np.sum(np.power(E_1-E_2,2))/np.sum(np.power(E_1-np.mean(E_1),2))
    text = f"MAE: {mae:.2f} eV\nPearson: {pearson:.4f}\nR2: {r2:.4f}"
    box_options = {
        'transform': ax.transAxes,
        'ha': 'left',
        'va': 'top',
        'bbox': {
            'boxstyle': 'round',
            'facecolor': 'wheat',
            'alpha': 0.5,
            },
        'fontsize': 5,
    }
    ax.text(0.05, 0.95, text, **box_options)
    n_data = len(E_1)
    title = f"{fig_title} ({n_data} rxns)"
    ax.set_title(title)

    save_options = {
        'dpi': 400,
        'bbox_inches': 'tight',
    }
    plt.savefig(f"rxn_E_{fig_title}_{cal_type_1}_{cal_type_2}.png", **save_options)
    print("done")


def plot_total(cal_type_1, cal_type_2):
    fig_title = 'Total'
    data_total = []
    for ion_type in ['CF', 'CF3', 'CH2F']:
        for ion_E in [20, 50]:
            src = f"/data2/andynn/ZBL_modify/SmallCell/05_reaction/{ion_type}/{ion_E}"
            data = get_data(src, cal_type_1, cal_type_2)
            if not data_total:
                data_total = data
            else:
                data_total = [np.concatenate((data_total[i], data[i])) for i in range(2)]
    plot(data_total, cal_type_1, cal_type_2, fig_title)


def main():
    if len(sys.argv) != 5:
        print("Usage: python cal_rxn_E.py src cal_type_1 cal_type_2 fig_title")
        sys.exit(1)
    src, cal_type_1, cal_type_2, fig_title = sys.argv[1:5]
    # data = get_data(src, cal_type_1, cal_type_2)
    # plot(data, cal_type_1, cal_type_2, fig_title)

    plot_total(cal_type_1, cal_type_2)


if __name__ == "__main__":
    main()
