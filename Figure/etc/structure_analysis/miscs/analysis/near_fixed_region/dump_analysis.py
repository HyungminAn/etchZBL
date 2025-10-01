import sys

from ase.io import read
from ase.geometry import get_distances
import matplotlib.pyplot as plt
import numpy as np


def make_pos_matrix(dump, id_to_idx_map):
    '''
    From dump, make position matrix which has shape of (len_atom, len_dump, 3)
    '''
    len_dump = len(dump)
    len_atom = len(id_to_idx_map)
    pos_matrix = np.zeros((len_atom, len_dump, 3))
    for i, image in enumerate(dump):
        atom_id = np.array([id_to_idx_map[i] for i in image.get_array('id')])
        pos_matrix[atom_id, i, :] = image.get_positions()
        print(f"Step {i}")
    return pos_matrix


def read_timestep(src_thermo):
    '''
    Read timestep from thermo file
    '''
    matrix = np.loadtxt(src_thermo, skiprows=2, usecols=(0, 1))
    _, unique_indices = np.unique(matrix[:, 0], return_index=True)
    reduced_matrix = matrix[unique_indices]
    return reduced_matrix


def get_disp_matrix(pos_matrix, cell):
    '''
    Get displacement matrix from position matrix (selected atoms)
    '''
    len_atom, len_dump, _ = pos_matrix.shape
    disp_matrix = np.zeros((len_atom, len_dump))
    for step, step_prev in zip(range(1, len_dump), range(len_dump-1)):
        xyz = np.squeeze(pos_matrix[:, step, :])
        xyz_prev = np.squeeze(pos_matrix[:, step_prev, :])
        _, D_len = get_distances(xyz, p2=xyz_prev, pbc=True, cell=cell)
        disp = np.diag(D_len)
        disp_matrix[:, step] = disp
    return disp_matrix


def get_speed_matrix(disp_matrix, timestep):
    '''
    Get velocity matrix from position matrix (selected atoms)
    '''
    len_atom, len_dump = disp_matrix.shape
    speed_matrix = np.zeros((len_atom, len_dump))

    for i in range(1, len_dump):
        time = timestep[i, 1] - timestep[i-1, 1]
        disp = disp_matrix[:, i]
        speed = disp / time
        speed_matrix[:, i] = speed
    return speed_matrix


def make_temperature_matrix(speed_matrix, mass_list):
    '''
    Make temperature matrix from speed matrix and mass list
    '''
    ANGPS_TO_MPS = 100
    AMU_TO_KG = 1.66053906660e-27
    J_TO_EV = 6.242e+18
    KB = 8.617333262145e-5  # eV/K

    len_atom, len_dump = speed_matrix.shape
    temp_matrix = np.zeros((len_atom, len_dump))
    for i in range(len_dump):
        speed = speed_matrix[:, i]
        kE = 0.5 * (mass_list * AMU_TO_KG) * (speed * ANGPS_TO_MPS) **2
        temp = (2 * kE * J_TO_EV) / (3 * KB)
        temp_matrix[:, i] = temp
    return temp_matrix


def select_idx(h_matrix, fix_h):
    '''
    Select atoms whose Z-coordinate is fixed at the end of simulation
    '''
    len_atom = h_matrix.shape[0]
    z_list = []
    for i in range(len_atom):
        z = h_matrix[i, :].flatten()
        is_free_atom_at_first = z[0] > fix_h
        # is_fixed_atom_at_last = z[-1] <= fix_h
        is_fixed_atom_at_last = z[-1] <= fix_h * 2
        if is_free_atom_at_first and is_fixed_atom_at_last:
            z_list.append(i)
    return np.array(z_list)


def make_mass_list(dump, id_to_idx_map):
    len_atom = len(id_to_idx_map)
    mass_list = np.zeros(len_atom)
    for image in dump:
        mass = image.get_masses()
        atom_id = image.get_array('id')

        for i, m in zip(atom_id, mass):
            mass_list[id_to_idx_map[i]] = m
    return mass_list


def make_id_to_idx_map(dump):
    id_set = set()
    for image in dump:
        id_set.update(image.get_array('id'))
    return { k: i for i, k in enumerate(id_set) }


def get_data(src, src_thermo, fix_h):
    timestep = read_timestep(src_thermo)
    dump = read(src, index=":")
    id_to_idx_map = make_id_to_idx_map(dump)
    pos_matrix = make_pos_matrix(dump, id_to_idx_map)
    idx_selected = select_idx(pos_matrix[:, :, 2], fix_h)
    pos_matrix = pos_matrix[idx_selected, :, :]

    cell = dump[0].get_cell()
    cell_z = cell[2, 2]
    disp_matrix = get_disp_matrix(pos_matrix, cell)
    speed_matrix = get_speed_matrix(disp_matrix, timestep)
    mass_list = make_mass_list(dump, id_to_idx_map)
    mass_list = mass_list[idx_selected]
    temp_matrix = make_temperature_matrix(speed_matrix, mass_list)

    result = {
            "timestep": timestep,
            "pos_matrix": pos_matrix,
            "temp_matrix": temp_matrix,
            "cell_z": cell_z,
            }

    return result


def plot(result, fix_h, subax=False):
    '''
    Plot Z-coordinate of selected atoms
    '''
    timestep = result["timestep"]
    pos_matrix = result["pos_matrix"]
    temp_matrix = result["temp_matrix"]
    cell_z = result["cell_z"]

    fig, (ax_z, ax_temp) = plt.subplots(2, 1, figsize=(6, 8))
    if subax:
        ax_z_sub = add_subplot_axes(ax_z, [0.2, 0.2, 0.4, 0.4])
    h_matrix = pos_matrix[:, :, 2]
    len_atom = h_matrix.shape[0]
    x = timestep[:, 1]
    for i in range(len_atom):
        z = h_matrix[i, :].flatten()
        temp = temp_matrix[i, :]
        ax_z.plot(x, z, alpha=0.3)
        ax_temp.plot(x, temp, alpha=0.3)
        if subax:
            ax_z_sub.plot(x, z)

    temp_sum = np.sum(temp_matrix, axis=0) / len_atom
    ax_temp.plot(x, temp_sum, color='black', linestyle='--', label="Total temperature")

    ax_z.set_xlabel("Time (ps)")
    ax_z.set_ylabel("Z-coordinate")
    ax_z.set_title("Z-coordinate of each atom")
    # ax_z.set_ylim(0, cell_z)
    ax_z.set_ylim(0, None)
    ax_z.axhline(y=fix_h,
                 color='grey',
                 linestyle='--',
                 label="Fix h")
    ax_z.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    if subax:
        ax_z_sub.set_ylim(0, cell_z)
        ax_z_sub.axhline(y=fix_h,
                         color='grey',
                         linestyle='--',
                         label="Fix h")

    ax_temp.set_xlabel("Time (ps)")
    ax_temp.set_ylabel("Temperature (K)")
    ax_temp.set_ylim(0, 2000)
    ax_temp.axhline(y=350,
                    color='white',
                    linestyle='--',
                    label="NVT temperature")
    ax_temp.axvline(x=2.0,
                    color='white',
                    linestyle='--',
                    label="NVT start")
    ax_temp.set_title("Temperature of each atom")
    ax_temp.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    fig.tight_layout()
    fig.savefig("Z-coordinate.png")


def add_subplot_axes(ax, rect, facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def main(src_thermo, src):
    fix_h = 6.0
    result = get_data(src, src_thermo, fix_h)
    plot(result, fix_h)


if __name__ == "__main__":
    src_thermo = sys.argv[1]
    src = sys.argv[2]
    main(src_thermo, src)
