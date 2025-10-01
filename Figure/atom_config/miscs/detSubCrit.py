import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt


def get_data(merged_cells_with_unitcell):
    def get_elem_count(slab, min_h, max_h):
        elem_count = {}
        elem_list = ['Si', 'O', 'C', 'H', 'F']
        symbols = slab.get_chemical_symbols()
        pos = slab.get_positions()
        for elem in elem_list:
            mask = np.array([True
                             if s == elem and xyz[2] >= min_h and xyz[2] <= max_h
                             else False
                             for s, xyz in zip(symbols, pos)])
            elem_count[elem] = np.sum(mask)
        return elem_count

    result = {
            'elem_count_region_1': {},  # fixed region
            'elem_count_region_2': {},  # next-to-fixed region
            'elem_count_region_3': {},  # region for CFx-ratio check
            'elem_count_total': {},     # total count
            'fix_z': {},
            }
    atoms_dict = merged_cells_with_unitcell['atoms']
    fix_h_dict = merged_cells_with_unitcell['fix_height']
    shift = 6.0
    for key, value in atoms_dict.items():
        fix_h = fix_h_dict[key]
        result['fix_z'][key] = fix_h

        min_h, max_h = fix_h - shift, fix_h
        elem_count_region_1 = get_elem_count(value, min_h, max_h)
        result['elem_count_region_1'][key] = elem_count_region_1

        min_h, max_h = fix_h, fix_h + shift
        elem_count_region_2 = get_elem_count(value, min_h, max_h)
        result['elem_count_region_2'][key] = elem_count_region_2

        min_h, max_h = fix_h + shift, np.inf
        elem_count_region_3 = get_elem_count(value, min_h, max_h)
        result['elem_count_region_3'][key] = elem_count_region_3

        min_h, max_h = fix_h - shift, np.inf
        elem_count_total = get_elem_count(value, min_h, max_h)
        result['elem_count_total'][key] = elem_count_total

        print(f"{key} Done")
    return result


def plot(result):
    fig, ax_mat = plt.subplots(4, 2, figsize=(12, 12))

    ax_region_1_count, ax_region_1_ratio = ax_mat[0]
    ax_region_2_count, ax_region_2_ratio = ax_mat[1]
    ax_region_3_count, ax_region_3_ratio = ax_mat[2]
    ax_total_count, ax_total_ratio = ax_mat[3]

    data_region_1 = result['elem_count_region_1']
    data_region_2 = result['elem_count_region_2']
    data_region_3 = result['elem_count_region_3']
    data_total = result['elem_count_total']
    fix_z = result['fix_z']

    keys = sorted(data_total.keys())
    elems = [('Si', 0),
             ('O', 1),
             ('C', 2),
             ('H', 3),
             ('F', 4)]
    n_iter, n_elem = len(keys), len(elems)
    mat_region_1 = np.zeros((n_iter, n_elem))
    mat_region_2 = np.zeros((n_iter, n_elem))
    mat_region_3 = np.zeros((n_iter, n_elem))
    mat_total_count = np.zeros((n_iter, n_elem))
    for (elem, elem_idx) in elems:
        y_region_1 = np.array([data_region_1[key][elem] for key in keys])
        y_region_2 = np.array([data_region_2[key][elem] for key in keys])
        y_region_3 = np.array([data_region_3[key][elem] for key in keys])
        y_total_count = np.array([data_total[key][elem] for key in keys])

        mat_region_1[:, elem_idx] = y_region_1
        mat_region_2[:, elem_idx] = y_region_2
        mat_region_3[:, elem_idx] = y_region_3
        mat_total_count[:, elem_idx] = y_total_count

    mat_total_count = np.sum(mat_total_count, axis=1)

    mat_region_1_ratio = mat_region_1 / np.sum(mat_region_1, axis=1)[:, None]
    mat_region_2_ratio = mat_region_2 / np.sum(mat_region_2, axis=1)[:, None]
    mat_region_3_ratio = mat_region_3 / np.sum(mat_region_3, axis=1)[:, None]
    mat_total_count_ratio = mat_total_count / mat_total_count[0]

    x = keys
    ax_total_count.plot(x, mat_total_count, label='total count')
    ax_total_ratio.plot(x, mat_total_count_ratio, label='total count')
    for (elem, elem_idx) in elems:
        ax_region_1_count.plot(x, mat_region_1[:, elem_idx], label=elem)
        ax_region_2_count.plot(x, mat_region_2[:, elem_idx], label=elem)
        ax_region_3_count.plot(x, mat_region_3[:, elem_idx], label=elem)

        ax_region_1_ratio.plot(x, mat_region_1_ratio[:, elem_idx], label=elem)
        ax_region_2_ratio.plot(x, mat_region_2_ratio[:, elem_idx], label=elem)
        ax_region_3_ratio.plot(x, mat_region_3_ratio[:, elem_idx], label=elem)

    # CFx-ratio check
    y = mat_region_1_ratio[:, 2] + mat_region_1_ratio[:, 4]
    ax_region_1_ratio.plot(x, y, label='C+F', linestyle='--', color='black')
    y = mat_region_2_ratio[:, 2] + mat_region_2_ratio[:, 4]
    ax_region_2_ratio.plot(x, y, label='C+F', linestyle='--', color='black')
    y = mat_region_3_ratio[:, 2] + mat_region_3_ratio[:, 4]
    ax_region_3_ratio.plot(x, y, label='C+F', linestyle='--', color='black')

    changed_idx = [key for key, key_prev in zip(keys[1:], keys[:-1]) if fix_z[key] != fix_z[key_prev]]
    for idx in changed_idx:
        ax_region_1_count.axvline(x=idx, color='gray', linestyle='--')
        ax_region_2_count.axvline(x=idx, color='gray', linestyle='--')
        ax_region_3_count.axvline(x=idx, color='gray', linestyle='--')
        ax_total_count.axvline(x=idx, color='gray', linestyle='--')

        ax_region_1_ratio.axvline(x=idx, color='gray', linestyle='--')
        ax_region_2_ratio.axvline(x=idx, color='gray', linestyle='--')
        ax_region_3_ratio.axvline(x=idx, color='gray', linestyle='--')
        ax_total_ratio.axvline(x=idx, color='gray', linestyle='--')

    ax_region_1_count.set_title('Region 1 Count')
    ax_region_2_count.set_title('Region 2 Count')
    ax_region_3_count.set_title('Region 3 Count')
    ax_total_count.set_title('Total Count')
    ax_region_1_ratio.set_title('Region 1 Ratio')
    ax_region_2_ratio.set_title('Region 2 Ratio')
    ax_region_3_ratio.set_title('Region 3 Ratio')
    ax_total_ratio.set_title('Total Ratio')

    ax_region_1_count.legend(loc='upper left')
    ax_region_2_count.legend(loc='upper left')
    ax_region_3_count.legend(loc='upper left')
    ax_total_count.legend(loc='upper left')
    ax_region_1_ratio.legend(loc='upper left')
    ax_region_2_ratio.legend(loc='upper left')
    ax_region_3_ratio.legend(loc='upper left')
    ax_total_ratio.legend(loc='upper left')

    ax_region_1_count.set_xlabel('Incidence')
    ax_region_2_count.set_xlabel('Incidence')
    ax_region_3_count.set_xlabel('Incidence')
    ax_total_count.set_xlabel('Incidence')
    ax_region_1_ratio.set_xlabel('Incidence')
    ax_region_2_ratio.set_xlabel('Incidence')
    ax_region_3_ratio.set_xlabel('Incidence')
    ax_total_ratio.set_xlabel('Incidence')

    ax_region_1_count.set_ylabel('Count')
    ax_region_2_count.set_ylabel('Count')
    ax_region_3_count.set_ylabel('Count')
    ax_total_count.set_ylabel('Count')
    ax_region_1_ratio.set_ylabel('Ratio')
    ax_region_2_ratio.set_ylabel('Ratio')
    ax_region_3_ratio.set_ylabel('Ratio')
    ax_total_ratio.set_ylabel('Ratio')

    fig.tight_layout()
    fig.savefig('result.png')


def main():
    if len(sys.argv) < 2:
        print("Usage: python detSubCrit.py <MergedCellsWithUnitCell.pkl>")
    path_pkl = sys.argv[1]
    with open(path_pkl, 'rb') as f:
        merged_cells_with_unitcell = pickle.load(f)

    result = get_data(merged_cells_with_unitcell)
    plot(result)


if __name__ == '__main__':
    main()
