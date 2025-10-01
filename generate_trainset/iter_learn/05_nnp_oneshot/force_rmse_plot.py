from ase.io import read
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tempfile import NamedTemporaryFile as NTF
import pickle
import os
from sklearn.metrics import mean_squared_error as rmse
import matplotlib.pyplot as plt
import numpy as np


def compress_outcar(filename, res):
    """
    *** From SIMPLE-NN code ***

    Compress VASP OUTCAR file for fast file-reading in ASE.
    Compressed file (tmp_comp_OUTCAR)
        is temporarily created in the current directory.

    :param str filename: filename of OUTCAR

    supported properties:

    - atom types
    - lattice vector(cell)
    - free energy
    - force
    - stress
    """

    with open(filename, 'r') as fil:
        minus_tag = 0
        line_tag = 0
        ions_key = 0
        for line in fil:
            if 'POTCAR:' in line:
                res.write(line)
            if 'POSCAR:' in line:
                res.write(line)
            elif 'ions per type' in line and ions_key == 0:
                res.write(line)
                ions_key = 1
            elif 'direct lattice vectors' in line:
                res.write(line)
                minus_tag = 3
            elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
                res.write(line)
                minus_tag = 4
            elif 'POSITION          ' in line:
                res.write(line)
                line_tag = 3
            elif 'FORCE on cell =-STRESS' in line:
                res.write(line)
                minus_tag = 15
            elif 'Iteration' in line:
                res.write(line)
            elif minus_tag > 0:
                res.write(line)
                minus_tag -= 1
            elif line_tag > 0:
                res.write(line)
                if '-------------------' in line:
                    line_tag -= 1


def read_force(path_dft, path_nnp):
    with NTF(
            mode='w+', encoding='utf-8', delete=True, prefix='OUTCAR_'
            ) as tmp_outcar:
        compress_outcar(path_dft, tmp_outcar)
        tmp_outcar.flush()
        tmp_outcar.seek(0)
        path_dft_compressed = tmp_outcar.name
        f_dft = read(path_dft_compressed).get_forces(apply_constraint=False)
    f_nnp = read(path_nnp).get_forces(apply_constraint=False)
    print(f'{path_dft} complete')
    return f_dft, f_nnp


def read_force_all(path_dft_list, path_nnp_list):
    with ThreadPoolExecutor() as executor:
        futures = []

        for path_dft, path_nnp in zip(path_dft_list, path_nnp_list):
            futures.append(
                executor.submit(
                    read_force,
                    path_dft,
                    path_nnp
                )
            )

        return [future.result() for future in as_completed(futures)]


def draw_parity_plot_force(ax, f_tot, idx_start, idx_end, ion_type):
    f_dft = [i[0].flatten() for i in f_tot[idx_start:idx_end]]
    f_dft = np.concatenate(f_dft)
    f_nnp = [i[1].flatten() for i in f_tot[idx_start:idx_end]]
    f_nnp = np.concatenate(f_nnp)
    f_rmse = rmse(f_dft, f_nnp, squared=False)

    ax.scatter(f_dft, f_nnp, alpha=0.3)
    ax.set_title(ion_type)
    ax.set_xlabel('DFT force (eV/A)')
    ax.set_ylabel('NNP force (eV/A)')

    ax.axline((0, 0), slope=1, linestyle='--', color='grey')
    max_value = max(
            max(f_dft.min(), f_dft.max(), key=abs),
            max(f_nnp.min(), f_nnp.max(), key=abs),
            )
    ax.set_xlim(-max_value, max_value)
    ax.set_ylim(-max_value, max_value)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textbox = f"Force RMSE : {f_rmse:.3f} (eV/A)"
    ax.text(0.95, 0.05, textbox,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes, bbox=props)


def plot_all(f_tot, ion_type_list):
    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten(order='F')

    idx_start = 0
    shift = 250
    idx_end = idx_start + shift

    for ax, ion_type in zip(axes, ion_type_list):
        draw_parity_plot_force(ax, f_tot, idx_start, idx_end, ion_type)
        idx_start += shift
        idx_end += shift
        print(f'{ion_type} Done')

    fig.tight_layout()
    fig.savefig('Traj_Force_RMSE.png')


def main():
    path_lammps_outs = 'lammps_outs'
    path_dft_oneshot = '../../03_dft_oneshot'

    ion_type_list = [
        'CF_10',
        'CF_30',
        'CF3_10',
        'CF3_30',
        'CH2F_10',
        'CH2F_30',
        'CHF2_10',
        'CHF2_30',
    ]

    path_dft_list = [
        f'{path_dft_oneshot}/{ion_type}/POSCAR_{i}_{j}/OUTCAR'
        for ion_type in ion_type_list
        for i in range(1, 51)
        for j in range(5)
    ]

    path_nnp_list = [
        f'{path_lammps_outs}/dump_{i}.lammpstrj'
        for i in range(1, 2001)
    ]

    path_force_dat = 'force_total.dat'
    if os.path.isfile(path_force_dat):
        with open(path_force_dat, 'rb') as f:
            f_tot = pickle.load(f)
    else:
        f_tot = read_force_all(path_dft_list, path_nnp_list)
        with open(path_force_dat, 'wb') as f:
            pickle.dump(f_tot, f)

    plot_all(f_tot, ion_type_list)


main()
