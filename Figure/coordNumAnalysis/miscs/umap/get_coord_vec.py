import sys
import time
from dataclasses import dataclass
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import NeighborList
# from scipy.spatial import cKDTree

# Global timing storage
_time_records = {}

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

class CoordVecGenerator:
    """
    Generates Gaussian-smoothed coordination vectors for each atom,
    split by central-element.
    """
    def __init__(self, params: PARAMS):
        self.params = params
        self.bin_centers = np.arange(
            params.R_MIN + params.BIN_WIDTH/2,
            params.R_MAX,
            params.BIN_WIDTH
        )
        self.n_bins = len(self.bin_centers)
        self.n_elem = len(params.ELEM_LIST)
        self.nbins_total = self.n_bins * self.n_elem

    @timing
    def load_cutoff_matrix(self, path: str) -> np.ndarray:
        """Load a 5×5 cutoff matrix from a text or NumPy file."""
        try:
            return np.loadtxt(path)
        except Exception:
            return np.load(path)

    @timing
    def setup_neighbor_list(self, atoms, cutoff_mat: np.ndarray) -> NeighborList:
        """Initialize ASE NeighborList with uniform radius = max_cutoff/2."""
        max_cut = np.max(cutoff_mat)
        radii = [max_cut/2] * len(atoms)
        nl = NeighborList(radii, self_interaction=False, bothways=True)
        nl.update(atoms)
        return nl

    @staticmethod
    @timing
    def build_coord_vec(distances: np.ndarray, bin_centers: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian-smooth each neighbor histogram row then sum columns."""
        diff = distances[:, None] - bin_centers[None, :]
        raw = np.exp(-0.5 * (diff / sigma)**2)
        normed = raw / (raw.sum(axis=1, keepdims=True) + 1e-12)
        return normed.sum(axis=0)

    @timing
    def load_image(self, path: str, format) -> np.ndarray:
        """Load an atoms image from a file."""
        if format == 'lammps':
            return read(path, **self.params.LAMMPS_READ_OPTS)
        elif format == 'vasp':
            return read(path, **self.params.VASP_READ_OPTS)
        else:
            raise ValueError(f"Unsupported format: {format}")

    # @timing
    # def find_neighbors_ckdtree(self, positions, cell, cutoff):
    #     """
    #     positions : (N,3) array of atom positions
    #     cell      : (3,3) periodic cell vectors (rows or cols)
    #     cutoff    : scalar distance cutoff

    #     Returns:
    #       neighbors : list of arrays of neighbor indices for each atom
    #       offsets   : list of arrays of periodic-image offsets (i,j,k) for each neighbor
    #     """
    #     # 1) build the 27-image point set
    #     #    We'll create image shifts in [-1,0,1]^3
    #     shifts = np.array([[i,j,k] for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)])
    #     images = (positions[None,:, :] + shifts[:,None,:].dot(cell)).reshape(-1,3)

    #     # 2) build KDTree on the expanded set
    #     tree = cKDTree(images)

    #     # 3) query each original point for neighbors within cutoff
    #     N = len(positions)
    #     neighbors = []
    #     offsets   = []

    #     for i in range(N):
    #         pt = positions[i]
    #         # find all image‐points within R
    #         idxs = tree.query_ball_point(pt, cutoff)
    #         # translate those back to (atom_index, shift)
    #         nbrs = []
    #         offs = []
    #         for idx in idxs:
    #             img_id = idx // N        # which shift
    #             atom_j = idx % N         # which original atom
    #             shift3 = shifts[img_id]  # e.g. [-1, 0, 1] triad
    #             nbrs.append(atom_j)
    #             offs.append(shift3)
    #         neighbors.append(np.array(nbrs, dtype=int))
    #         offsets.append(np.array(offs, dtype=int))

    #     return neighbors, offsets

    @timing
    def compute(self, structure_path: str, cutoff_path: str, format=None) -> dict:
        """
        Read structure, load cutoff matrix, compute and return
        per-element coordination data dict.
        """
        atoms = self.load_image(structure_path, format=format)
        cutoff_mat = self.load_cutoff_matrix(cutoff_path)
        nl = self.setup_neighbor_list(atoms, cutoff_mat)

        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)
        full_feat = np.zeros((n_atoms, self.nbins_total))

        for i in range(n_atoms):
            neigh, offs = nl.get_neighbors(i)
            # neigh, offs = self.find_neighbors_ckdtree(atoms.get_positions(),
            #                                           atoms.get_cell(),
            #                                           cutoff_mat.max())
            if len(neigh) == 0:
                continue
            dists_per = {el: [] for el in self.params.ELEM_LIST}
            for j, off in zip(neigh, offs):
                # vec = atoms[j].position + offs and np.dot(off, atoms.get_cell()) or 0 - atoms[i].position
                vec = atoms[j].position + np.dot(off, atoms.get_cell()) - atoms[i].position
                d = np.linalg.norm(vec)
                ei = symbols[i]; ej = symbols[j]
                idx_i = self.params.ELEM_LIST.index(ei)
                idx_j = self.params.ELEM_LIST.index(ej)
                if d <= cutoff_mat[idx_i, idx_j]:
                    dists_per[ej].append(d)
            for k, el in enumerate(self.params.ELEM_LIST):
                arr = np.array(dists_per[el])
                hist = self.build_coord_vec(arr, self.bin_centers, self.params.SIGMA) if arr.size else np.zeros(self.n_bins)
                full_feat[i, k*self.n_bins:(k+1)*self.n_bins] = hist

        coord_data = {}
        for k, el in enumerate(self.params.ELEM_LIST):
            mask = [i for i,s in enumerate(symbols) if s == el]
            coord_data[el] = full_feat[mask, :]
        return coord_data

    @timing
    def save(self, coord_data: dict):
        """Save each element’s matrix to .npy files."""
        for el, mat in coord_data.items():
            fname = f"{self.params.OUTPUT_PREFIX}_{el}.npy"
            np.save(fname, mat)

class CoordVecPlotter:
    """
    Plots distribution or cumsum of coordination vectors
    in a grid of central vs neighbor elements.
    """
    def __init__(self, params: PARAMS):
        self.params = params

    @timing
    def plot_distribution(self, coord_data: dict, log_scale=False):
        """Plot column-wise sums for each element pair in an N×N grid."""
        elems = self.params.ELEM_LIST; n = len(elems)
        n_bins = next(iter(coord_data.values())).shape[1] // n
        bins = np.arange(
            self.params.R_MIN + self.params.BIN_WIDTH/2,
            self.params.R_MAX,
            self.params.BIN_WIDTH
        )
        # find ymin for log
        ymin = None
        if log_scale:
            pos = []
            for mat in coord_data.values():
                for j in range(n): pos.extend(mat[:, j*n_bins:(j+1)*n_bins].sum(axis=0)[lambda x: x>0])
            ymin = (min(pos)/10) if pos else 1e-6

        plt.rcParams.update({'font.size': 12, 'font.family': 'arial'})
        fig, axes = plt.subplots(n, n, figsize=(3*n,3*n), squeeze=False)
        for i, cent in enumerate(elems):
            for j, nei in enumerate(elems):
                ax = axes[i,j]
                block = coord_data[cent][:, j*n_bins:(j+1)*n_bins]
                if not block.size or np.allclose(block,0): fig.delaxes(ax); continue
                col = block.sum(axis=0)
                ax.plot(bins, col, lw=2)
                if log_scale:
                    ax.set_yscale('log'); ax.set_ylim(bottom=ymin)
                ax.set_title(f"Center: {cent} → Neighbor: {nei}")
                ax.set_xlabel("Radial distance (Å)")
                ax.set_ylabel("Count" + (" (log)" if log_scale else ""))
        fig.tight_layout(); fig.savefig(f"{self.params.OUTPUT_PREFIX}_grid.png", dpi=300); plt.close(fig)

def print_timing_summary():
    total = sum(_time_records.values())
    print("Timing Summary:")
    for fn, t in sorted(_time_records.items(), key=lambda x: -x[1]):
        print(f"  {fn}: {t:.3f}s ({100*t/total:.1f}%)")

def main():
    if len(sys.argv)!=3:
        print("Usage: python get_coord_vec.py <structure> <cutoff.npy>")
        sys.exit(1)
    p = PARAMS()
    gen = CoordVecGenerator(p)
    data = gen.compute(sys.argv[1], sys.argv[2], format='lammps')
    gen.save(data)

    plotter = CoordVecPlotter(p)
    plotter.plot_distribution(data, log_scale=False)

    print_timing_summary()

if __name__ == '__main__':
    main()
