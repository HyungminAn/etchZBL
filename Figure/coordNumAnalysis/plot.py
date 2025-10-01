#!/usr/bin/env python3
import os
import sys
import time
import argparse
from dataclasses import dataclass
from functools import wraps
from math import ceil
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from ase.io import read
from ase.neighborlist import NeighborList
from ase.geometry import get_distances
from braceexpand import braceexpand
"""
python plot_coord_numbers.py <structure_list_file> <cutoff_matrix.npy>
    --method fractional/discrete
    --style  hist/curve
"""

# Global timing storage
_time_records = {}

@dataclass
class CovalentRadii:
    covalent_radii = np.array([
        0.00000000,
        0.80628308, 1.15903197, 3.02356173, 2.36845659, 1.94011865,
        1.88972601, 1.78894056, 1.58736983, 1.61256616, 1.68815527,
        3.52748848, 3.14954334, 2.84718717, 2.62041997, 2.77159820,
        2.57002732, 2.49443835, 2.41884923, 4.43455700, 3.88023730,
        3.35111422, 3.07395437, 3.04875805, 2.77159820, 2.69600923,
        2.62041997, 2.51963467, 2.49443835, 2.54483100, 2.74640188,
        2.82199085, 2.74640188, 2.89757982, 2.77159820, 2.87238349,
        2.94797246, 4.76210950, 4.20778980, 3.70386304, 3.50229216,
        3.32591790, 3.12434702, 2.89757982, 2.84718717, 2.84718717,
        2.72120556, 2.89757982, 3.09915070, 3.22513231, 3.17473967,
        3.17473967, 3.09915070, 3.32591790, 3.30072128, 5.26603625,
        4.43455700, 4.08180818, 3.70386304, 3.98102289, 3.95582657,
        3.93062995, 3.90543362, 3.80464833, 3.82984466, 3.80464833,
        3.77945201, 3.75425569, 3.75425569, 3.72905937, 3.85504098,
        3.67866672, 3.45189952, 3.30072128, 3.09915070, 2.97316878,
        2.92277614, 2.79679452, 2.82199085, 2.84718717, 3.32591790,
        3.27552496, 3.27552496, 3.42670319, 3.30072128, 3.47709584,
        3.57788113, 5.06446567, 4.56053862, 4.20778980, 3.98102289,
        3.82984466, 3.85504098, 3.88023730, 3.90543362,
    ]) * 0.52917726

def timing(func):
    """
    Decorator to time methods and store results in _time_records.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        _time_records[func.__name__] = _time_records.get(func.__name__, 0) + elapsed
        return result
    return wrapper

@dataclass
class PARAMS:
    # Radial histogram parameters
    BIN_WIDTH: float = 0.2    # Å
    R_MIN:    float = 0.5     # Å
    R_MAX:    float = 6.0     # Å
    SIGMA:    float = 0.5     # Å, Gaussian width

    # Element information
    ELEM_LIST = ['Si', 'O', 'C', 'H', 'F']
    OUTPUT_PREFIX: str = 'coord_vecs'

    # LAMMPS read options for ASE
    ATOM_IDX_Si, ATOM_NUM_Si = 1, 14
    ATOM_IDX_O,  ATOM_NUM_O  = 2, 8
    ATOM_IDX_C,  ATOM_NUM_C  = 3, 6
    ATOM_IDX_H,  ATOM_NUM_H  = 4, 1
    ATOM_IDX_F,  ATOM_NUM_F  = 5, 9
    LAMMPS_READ_OPTS = {
        'format': 'lammps-data',
        'Z_of_type': {
            ATOM_IDX_Si: ATOM_NUM_Si,
            ATOM_IDX_O:  ATOM_NUM_O,
            ATOM_IDX_C:  ATOM_NUM_C,
            ATOM_IDX_H:  ATOM_NUM_H,
            ATOM_IDX_F:  ATOM_NUM_F,
        }
    }
    VASP_READ_OPTS = {
        'format': 'vasp-out',
    }
    COLOR_DICT = {
        'low_density': 'red',
        'normal_density': 'green',
        }

class DiscreteCoordinationCalculator:
    """
    Compute discrete coordination numbers (integer counts) using fixed cutoff matrix.
    """
    def compute(self, atoms, cutoff_mat, elem_list):
        # build neighbor list with uniform radius = max_cutoff/2
        max_cut = np.max(cutoff_mat)
        nl = NeighborList([max_cut/2]*len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)

        symbols = atoms.get_chemical_symbols()
        coord_numbers = {el: [] for el in elem_list}

        for i, sym_i in enumerate(symbols):
            neigh_inds, offsets = nl.get_neighbors(i)
            count = 0
            pos_i = atoms[i].position
            cell = atoms.get_cell()
            idx_i = elem_list.index(sym_i)
            for j, off in zip(neigh_inds, offsets):
                disp = atoms[j].position + np.dot(off, cell) - pos_i
                d = np.linalg.norm(disp)
                idx_j = elem_list.index(symbols[j])
                if d <= cutoff_mat[idx_i, idx_j]:
                    count += 1
            coord_numbers[sym_i].append(count)

        return coord_numbers

class FractionalCoordinationCalculator:
    """
    Compute fractional coordination numbers using smooth switching function from coordnum.py.
    """
    def __init__(self, k1=16.0):
        self.k1 = k1

    def compute(self, atoms, cutoff_mat, elem_list):
        # positions, cell, atomic numbers
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        symbols = atoms.get_chemical_symbols()
        atomic_numbers = atoms.get_atomic_numbers()
        # compute distance matrix
        _, dist_mat = get_distances(pos, pbc=True, cell=cell)
        # covalent radii lookup
        cr = CovalentRadii.covalent_radii
        radii = np.array([cr[n] for n in atomic_numbers])
        k1 = self.k1
        # compute switching function
        with np.errstate(divide='ignore', invalid='ignore'):
            CN_inv = 1.0 + np.exp(-k1 * ((radii[:, None] + radii[None, :]) / dist_mat - 1.0))
            CN_mat = 1.0 / CN_inv
        # zero self interactions
        np.fill_diagonal(CN_mat, 0.0)
        # sum over neighbors
        CN_total = np.sum(CN_mat, axis=1)
        # organize by element
        coord_numbers = {el: [] for el in elem_list}
        for i, sym in enumerate(symbols):
            if sym in coord_numbers:
                coord_numbers[sym].append(CN_total[i])
        return coord_numbers

@timing
def parse_structure_list(filename):
    """
    Parse a file of the form:

        [class_name]
        /path/to/POSCAR_{0..9960..80}/OUTCAR :
        ...

    and return a dict: { class_name: [expanded_paths, ...], ... }
    """
    result = {}
    current = None
    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1]
                result[current] = []
                continue
            if current is None:
                continue
            if ':' in line:
                tpl, slice_txt = line.rsplit(':', 1)
                tpl = tpl.strip()
                slc = slice(None)
                if slice_txt.strip():
                    slc = eval(f"slice({slice_txt})")
            else:
                tpl, slc = line, slice(None)
            expanded = list(braceexpand(tpl))
            result[current].extend(expanded[slc])
    return result

@timing
def batch_calculate_coordnum(mapping, elem_list, cutoff_mat, fmt, method='discrete'):
    """
    mapping: dict[tag] -> list of file paths
    elem_list: list of element symbols
    cutoff_mat: (N_elem, N_elem) array for discrete method
    fmt: file format for ASE read
    method: 'discrete' or 'fractional'
    returns: dict[tag] -> dict[element] -> list of coordination numbers
    """
    path_save = "coordination_numbers.pkl"
    if os.path.exists(path_save):
        print(f"Loading existing coordination numbers from {path_save}…")
        with open(path_save, 'rb') as f:
            return pickle.load(f)

    coord_by_tag = {}
    log_step = 10
    for tag, paths in mapping.items():
        path_save_sub = f"{PARAMS.OUTPUT_PREFIX}_{tag}.pkl"
        if os.path.exists(path_save_sub):
            print(f"Loading existing coordination numbers for tag '{tag}' from {path_save_sub}…")
            with open(path_save_sub, 'rb') as f:
                coord_by_tag[tag] = pickle.load(f)
            continue

        all_co = {el: [] for el in elem_list}
        print(f"Processing tag '{tag}' ({len(paths)} structures)…")
        for idx, path in enumerate(paths):
            try:
                atoms = read(path, format=fmt)
            except Exception as e:
                print(f"  [WARN] cannot read {path}: {e}")
                continue
            if method == 'discrete':
                cn_dict = DiscreteCoordinationCalculator().compute(atoms, cutoff_mat, elem_list)
            else:
                cn_dict = FractionalCoordinationCalculator().compute(atoms, cutoff_mat, elem_list)
            for el in elem_list:
                all_co[el].extend(cn_dict[el])

            if idx % log_step == log_step - 1:
                print(f"  processed {idx+1}/{len(paths)} structures; path: {path}")

        coord_by_tag[tag] = all_co
        print(f"  saving coordination numbers for tag '{tag}' to {path_save_sub}…")
        with open(path_save_sub, 'wb') as f:
            pickle.dump(all_co, f)

    with open(path_save, 'wb') as f:
        pickle.dump(coord_by_tag, f)

    return coord_by_tag

@timing
def plot(elem_list, coord_by_tag,
         groups=None, exclude_tags=None,
         style='hist', out_png="result.png"):
    """
    elem_list: list of element symbols
    coord_by_tag: dict[tag] -> dict[element] -> list of coordination numbers (int or float)
    groups: dict[group_name] -> list of tags to combine (optional)
    exclude_tags: list of tags to skip (optional)
    style: 'hist' for stacked histogram, 'curve' for shaded KDE curves
    out_png: filename for saving the figure
    """
    # Filter tags
    all_tags = list(coord_by_tag.keys())
    tags_filtered = [t for t in all_tags if not exclude_tags or t not in exclude_tags]

    # Determine plot labels
    plot_keys = list(groups.keys()) if groups else tags_filtered

    n_elem = len(elem_list)
    ncols = 3
    nrows = ceil(n_elem / ncols)
    plt.rcParams.update({'font.family': 'arial', 'font.size': 10})
    fig, axes = plt.subplots(nrows,
                             ncols,
                             # figsize=(3*ncols, 3*nrows),
                             figsize=(7.1, 5),
                             sharey=False)
    axes = axes.flatten()

    for i, el in enumerate(elem_list):
        ax = axes[i]
        # collect all values
        all_vals = []
        if groups:
            for grp in plot_keys:
                for t in groups[grp]:
                    if t in tags_filtered:
                        all_vals.extend(coord_by_tag[t][el])
        else:
            for t in tags_filtered:
                all_vals.extend(coord_by_tag[t][el])

        if not all_vals:
            ax.set_title(f"{el} (no data)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        max_cn = max(all_vals)
        bins = np.arange(-0.5, max_cn + 1.5, 0.2)

        if style == 'hist':
            data_list = []
            for key in plot_keys:
                merged = []
                members = groups[key] if groups else [key]
                for t in members:
                    if t in tags_filtered:
                        merged.extend(coord_by_tag[t][el])
                data_list.append(merged)
            ax.hist(data_list,
                    bins=bins,
                    histtype='stepfilled',
                    stacked=False,
                    density=False,
                    color=[PARAMS.COLOR_DICT.get(k) for k in plot_keys],
                    # edgecolor='black',
                    alpha=0.5)
        elif style == 'curve':
            x = np.linspace(0, max_cn, 200)
            for key in plot_keys:
                merged = []
                members = groups[key] if groups else [key]
                for t in members:
                    if t in tags_filtered:
                        merged.extend(coord_by_tag[t][el])
                data = merged
                if len(data) > 1:
                    kde = gaussian_kde(data)
                    y = kde(x)
                    ax.plot(x, y, label=key)
                    ax.fill_between(x, y, alpha=0.3)
        else:
            raise ValueError("style must be 'hist' or 'curve'")

        ax.set_title(el)
        ax.set_xlabel("Coordination number")
        ax.set_ylabel("Density" if style == 'curve' else "Counts")

    # remove unused axes
    for j in range(n_elem, nrows*ncols):
        fig.delaxes(axes[j])

    # Global legend
    legend_handles = [Patch(facecolor=PARAMS.COLOR_DICT.get(k), edgecolor='black') for k in plot_keys]
    fig.legend(legend_handles,
               plot_keys,
               loc='upper center',
               ncol=len(plot_keys),
               frameon=False,
               fontsize='small')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Compute and plot coordination numbers.")
    parser.add_argument('structure_list_file', help='file listing structures with tags')
    parser.add_argument('cutoff_file', help='cutoff matrix file (txt or npy)')
    parser.add_argument('--method', choices=['discrete', 'fractional'], default='discrete',
                        help='coordination calculation method')
    parser.add_argument('--fmt', default='vasp-out', help='ASE read format')
    parser.add_argument('--style', choices=['hist', 'curve'], default='hist', help='plot style')
    args = parser.parse_args()

    mapping = parse_structure_list(args.structure_list_file)
    try:
        cutoff_mat = np.loadtxt(args.cutoff_file)
    except:
        cutoff_mat = np.load(args.cutoff_file)

    elem_list = PARAMS.ELEM_LIST
    coord_by_tag = batch_calculate_coordnum(mapping, elem_list, cutoff_mat,
                                           fmt=args.fmt, method=args.method)

    plot_opts = {
        'groups': {
            'low_density': ['chf_low', 'vts_low'],
            # 'iterlearn': ['iterlearn'],
            'normal_density': ['chf_normal', 'vts_normal'],
        },
        'exclude_tags': ['bulk', 'slab', 'gas', 'chf_slab', 'iterlearn'],
        'style': args.style,
        'out_png': 'result.png'
    }
    plot(elem_list, coord_by_tag, **plot_opts)

if __name__ == "__main__":
    main()
