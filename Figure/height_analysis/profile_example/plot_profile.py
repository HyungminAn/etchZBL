import os
from functools import wraps
from dataclasses import dataclass

from ase.io import read
from ase.data import atomic_numbers, atomic_masses
from ase.neighborlist import NeighborList
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

@dataclass
class PARAMS:
    ATOM_IDX_Si, ATOM_NUM_Si = 1, 14
    ATOM_IDX_O, ATOM_NUM_O = 2, 8
    ATOM_IDX_C, ATOM_NUM_C = 3, 6
    ATOM_IDX_H, ATOM_NUM_H = 4, 1
    ATOM_IDX_F, ATOM_NUM_F = 5, 9

    LAMMPS_READ_OPTS = {
        'format': 'lammps-data',
        'Z_of_type': {
            ATOM_IDX_Si: ATOM_NUM_Si,
            ATOM_IDX_O: ATOM_NUM_O,
            ATOM_IDX_C: ATOM_NUM_C,
            ATOM_IDX_H: ATOM_NUM_H,
            ATOM_IDX_F: ATOM_NUM_F,
        }
    }

    ELEM_LIST = ['Si', 'O', 'C', 'H', 'F']

    NP_SAVE_OPTS = {
        'fmt': '%.3f %.3f',
        'header': 'key height(A)',
        'comments': ''
    }

    SPACING = 0.1
    BW_WIDTH = 0.2

    CUTOFF_RATIO_FILM = 0.6
    CUTOFF_RATIO_MIXED = 0.1
    CUTOFF_RATIO_FILM_UPPER = 0.5

    ATOM_COLOR = {
        'Si': '#F0C8A0',
        'O': '#FF0D0D',
        'N': '#3050F8',
        'C': '#909090',
        'H': '#FF00FF',
        'F': '#90E050',
    }


def save_npresult_as(func_gen_name):
    '''
    Decorator to save the result of a function as a numpy file.
    '''
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            path_save = func_gen_name(self)
            if os.path.exists(path_save):
                print(f"{path_save} already exists, loading data from it.")
                data = np.loadtxt(path_save, skiprows=1)
                x = data[:, 0].astype(float)
                y = data[:, 1]
                return x, y
            x, y = func(self, *args, **kwargs)
            np.savetxt(path_save, np.c_[x, y], **PARAMS.NP_SAVE_OPTS)
            return x, y
        return wrapper
    return decorator

class DensityProfileGenerator:
    def run(self, atoms):
        pos_z = atoms.get_positions()[:, 2]
        symbols = atoms.get_chemical_symbols()

        x = np.arange(0, atoms.get_cell()[2, 2], PARAMS.SPACING)

        result = {}
        result['z'] = x
        for elem in PARAMS.ELEM_LIST:
            mask = np.array([idx for idx, s in enumerate(symbols) if s == elem])
            if len(mask) <= 1:
                continue
            y = pos_z[mask]
            gaussian = gaussian_kde(y, bw_method=PARAMS.BW_WIDTH)
            density = gaussian.evaluate(x) * len(y)
            result[elem] = density
        return result

class DensityProfileAnalyzer:
    def run(self, atoms, profile):
        data = self.get_thickness(profile,
                                  PARAMS.CUTOFF_RATIO_FILM,
                                  PARAMS.CUTOFF_RATIO_MIXED,
                                  )
        density_mixed_layer = self.calculate_average_density(
                atoms, data['z_mixed_min'], data['z_mixed_max'])
        density_film_layer = self.calculate_average_density(
                atoms, data['z_film_min'], data['z_film_max'])
        fc_ratio_mixed_layer = self.calculate_FC_ratio(
                atoms, data['z_mixed_min'], data['z_mixed_max'])
        fc_ratio_film_layer = self.calculate_FC_ratio(
                atoms, data['z_film_min'], data['z_film_max'])

        spx_ratio = self.calculate_spx_ratio(
                atoms, data['z_film_min'], data['z_film_max'])

        carbon_status = self.get_carbon_status(atoms)

        data.update({
            'density_mixed_layer': density_mixed_layer,
            'density_film_layer': density_film_layer,
            'fc_ratio_mixed_layer': fc_ratio_mixed_layer,
            'fc_ratio_film_layer': fc_ratio_film_layer,
            'spx_ratio': spx_ratio,
        })

        self.print_results(data)

        return data

    def print_results(self, data):
        for key, val in data.items():
            if isinstance(val, float):
                print(f"{key:20}: {val:.3f}")
            elif isinstance(val, dict):
                print(f"{key:20}: {val}")
            elif val is None:
                print(f"{key:20}: None")
            else:
                print(f"{key:20}: {val}")
        return data

    def get_thickness(self,
                      profile,
                      cutoff_ratio_film,
                      cutoff_ratio_mixed):
        """
        profile: dict with keys 'z', 'Si', 'O', 'C', 'H', 'F'
        x1: peak threshold to define existence of film
        x2: cutoff for film thickness
        x3: cutoff for mixed layer thickness
        """
        z = profile['z']
        eps = 1e-8  # avoid divide by zero
        CHFs = profile.get('C', 0) + profile.get('H', 0) + profile.get('F', 0)
        SiOs = profile.get('Si', 0) + profile.get('O', 0) + eps
        total = CHFs + SiOs
        ratio = CHFs / total
        normalized_ratio = total / np.max(total)

        peak_val, flag_calculate_mixed, flag_calulate_film = \
            self.check_layer_status(ratio, cutoff_ratio_mixed, cutoff_ratio_film)

        z_mixed_min, z_mixed_max, h_mixed = \
            self.calculate_mixed_layer_thickness(flag_calculate_mixed,
                                                 z,
                                                 ratio,
                                                 cutoff_ratio_mixed,
                                                 cutoff_ratio_film)

        z_film_min, z_film_max, h_film = \
            self.calculate_film_thickness(flag_calulate_film,
                                          z,
                                          z_mixed_max,
                                          ratio,
                                          normalized_ratio,
                                          cutoff_ratio_film)

        result = {
                'z_mixed_min': z_mixed_min,
                'z_mixed_max': z_mixed_max,
                'h_mixed': h_mixed,
                'z_film_min': z_film_min,
                'z_film_max': z_film_max,
                'h_film': h_film,
                'z_max': np.max(ratio),
                'ratio_max': peak_val,
                }

        return result

    def check_layer_status(self, ratio, cutoff_ratio_mixed, cutoff_ratio_film):
        # Step 1: film exists?
        peak_val = np.max(ratio)
        if peak_val < cutoff_ratio_mixed:
            return peak_val, False, False
        elif cutoff_ratio_mixed < peak_val < cutoff_ratio_film:
            return peak_val, True, False
        elif cutoff_ratio_film < peak_val:
            return peak_val, True, True
        else:
            raise ValueError("Invalid cutoff values")

    def calculate_mixed_layer_thickness(self,
                                        flag_calculate_mixed,
                                        z,
                                        ratio,
                                        cutoff_ratio_mixed,
                                        cutoff_ratio_film):
        if not flag_calculate_mixed:
            return None, None, 0.0

        z_max = z[np.argmax(ratio)]
        mask_mixed = (z <= z_max) \
                     & (ratio >= cutoff_ratio_mixed) \
                     & (ratio < cutoff_ratio_film)
        if np.sum(mask_mixed) == 0:
            return None, None, 0.0

        z_mixed = z[mask_mixed]
        z_mixed_min = np.min(z_mixed)
        z_mixed_max = np.max(z_mixed)
        h_mixed = z_mixed_max - z_mixed_min  # mixed layer thickness
        return z_mixed_min, z_mixed_max, h_mixed

    def calculate_film_thickness(self,
                                 flag_calulate_film,
                                 z,
                                 z_mixed_max,
                                 ratio,
                                 normalized_ratio,
                                 cutoff_ratio_film):
        if not flag_calulate_film:
            return None, None, 0.0

        if z_mixed_max is None:
            z_mixed_max = 0.0

        mask_film = (z > z_mixed_max) \
                    & (ratio >= cutoff_ratio_film) \
                    & (normalized_ratio >= cutoff_ratio_film)
        if np.sum(mask_film) == 0:
            return None, None, 0.0

        z_film = z[mask_film]
        z_film_min = np.min(z_film)
        z_film_max = np.max(z_film)
        h_film = z_film_max - z_film_min

        return z_film_min, z_film_max, h_film

    def calculate_average_density(self, atoms, z_min, z_max):
        if z_min is None or z_max is None:
            return None

        cell = atoms.get_cell()
        pos_z = atoms.get_positions()[:, 2]
        mask = (np.logical_and(pos_z >= z_min, pos_z <= z_max))

        if np.sum(mask) == 0:
            return None

        symbols = atoms.get_chemical_symbols()
        masses = [atomic_masses[atomic_numbers[symbol]]
                  for idx, symbol in enumerate(symbols) if mask[idx]]
        masses = np.array(masses)
        total_mass = np.sum(masses)  # amu unit

        lat_x, lat_y = cell[0, 0], cell[1, 1]
        volume = lat_x * lat_y * (z_max - z_min)  # A^3

        AMU_TO_G = 1.66053906660e-24  # g
        A3_TO_CM3 = 1e-24  # cm^3

        density = (total_mass * AMU_TO_G) / (volume * A3_TO_CM3)  # g/cm^3
        return density

    def calculate_FC_ratio(self, atoms, z_min, z_max):
        if z_min is None or z_max is None:
            return None

        pos_z = atoms.get_positions()[:, 2]
        mask = (np.logical_and(pos_z >= z_min, pos_z <= z_max))

        if np.sum(mask) == 0:
            return None

        symbols = atoms.get_chemical_symbols()
        symbols = [symbols[idx] for idx in range(len(symbols)) if mask[idx]]
        mask_C = np.array([idx for idx, symbol in enumerate(symbols) if symbol == 'C'])
        mask_F = np.array([idx for idx, symbol in enumerate(symbols) if symbol == 'F'])
        if len(mask_C) ==0 or len(mask_F) == 0:
            return None
        fc_ratio = len(mask_F) / len(mask_C)
        return fc_ratio

    def calculate_spx_ratio(self, atoms, z_min, z_max):
        """
        Compute spx hybridization counts for carbon atoms within a specified z-range,
        accounting for periodic boundary conditions and per-pair cutoffs.

        Returns:
            dict: {'sp3': int, 'sp2': int, 'sp': int, 'others': int}
        """
        if z_min is None or z_max is None:
            return None

        # Positions and cell dimensions for PBC
        positions = atoms.get_positions()
        # Assumes orthorhombic cell: use diagonal entries
        box = atoms.get_cell().diagonal()
        tree = cKDTree(positions, boxsize=box)

        # Filter atoms by z-range
        z_coords = positions[:, 2]
        z_mask = (z_coords >= z_min) & (z_coords <= z_max)
        if not np.any(z_mask):
            return None

        symbols = atoms.get_chemical_symbols()
        # Map element to cutoff-matrix index
        symbol_map = {'Si': 0, 'O': 1, 'C': 2, 'H': 3, 'F': 4}

        # Load pairwise cutoff distances
        cutoff_mat = np.loadtxt('cutoff_matrix.npy')

        # Identify carbon atoms in region
        region_idx = np.where(z_mask)[0]
        is_carbon = np.array([symbols[i] == 'C' for i in region_idx])
        carbon_idx = region_idx[is_carbon]
        if carbon_idx.size == 0:
            return None

        # Build a tree for carbon atoms
        carbon_pos = positions[carbon_idx]
        carbon_tree = cKDTree(carbon_pos, boxsize=box)

        # Use sparse distance matrix to get all pairs within max cutoff
        max_cut = cutoff_mat.max()
        # sparse_distance_matrix returns dict {(i_local, j_global): distance}
        dist_dict = carbon_tree.sparse_distance_matrix(tree, max_cut)

        # Convert to arrays for vectorized filtering
        pairs = np.array(list(dist_dict.keys()), dtype=int)
        dists = np.array(list(dist_dict.values()))
        i_local, j_global = pairs[:, 0], pairs[:, 1]

        # Remove self-pairs
        global_i = carbon_idx[i_local]
        self_mask = (global_i == j_global)
        i_local = i_local[~self_mask]
        j_global = j_global[~self_mask]
        dists = dists[~self_mask]

        # Determine neighbor types and corresponding cutoffs
        j_types = np.array([symbol_map[symbols[j]] for j in j_global])
        c_type = symbol_map['C']
        valid_cut = cutoff_mat[c_type, j_types]
        valid = dists <= valid_cut

        # Count neighbors per carbon atom
        counts = np.bincount(i_local[valid], minlength=carbon_idx.size)

        # Classify hybridization
        spx_ratio = {
            'sp3': int(np.count_nonzero(counts == 4)),
            'sp2': int(np.count_nonzero(counts == 3)),
            'sp': int(np.count_nonzero(counts == 2)),
            'others': int(np.count_nonzero(~np.isin(counts, [2, 3, 4])))
        }

        # Debug output
        debug = True
        debug_n = 10
        if debug:
            print(f"Debug spx for first {min(debug_n, carbon_idx.size)} carbons:")
            for il in range(min(debug_n, carbon_idx.size)):
                gi = carbon_idx[il]
                cnt = int(counts[il]) if il < counts.size else 0
                hyb = ('sp3' if cnt == 4 else 'sp2' if cnt == 3 else 'sp' if cnt == 2 else 'others')
                print(f"Carbon global idx {gi} (local {il}): hybridization={hyb}, neighbor count={cnt}")
                mask_pair = (valid & (i_local == il))
                nbrs = j_global[mask_pair]
                dists_n = dists[mask_pair]
                for jg, dist in zip(nbrs, dists_n):
                    pos = positions[jg]
                    print(f"  Neighbor idx {jg}: pos={pos}, dist={dist:.3f}")

        return spx_ratio

    def get_carbon_status(self, atoms):
        cnc = CarbonNeighborClassifier(atoms)
        classification = cnc.run()

class CarbonNeighborClassifier:
    """
    Classify carbon atoms based on neighbor atom types using a cutoff matrix.

    Attributes:
        atoms: ASE Atoms object
        cutoff_matrix: 2D numpy array of pairwise cutoffs (element_type x element_type)
        element_order: list of element symbols defining indices in cutoff_matrix
    """
    def __init__(self, atoms, element_order=None):
        self.atoms = atoms
        self.cutoff_matrix = np.loadtxt('cutoff_matrix.npy')
        # Default element order if not provided
        if element_order is None:
            self.element_order = ["Si", "O", "C", "H", "F"]
        else:
            self.element_order = element_order
        self.symbol_to_index = {sym: i for i, sym in enumerate(self.element_order)}

    def run(self):
        """
        Classify each carbon atom based on its neighbors.
        Returns:
            dict: mapping carbon atom index -> classification string
        """
        adjacency = self.build_adjacency()
        symbols = self.atoms.get_chemical_symbols()
        classification = {}
        # Iterate over all atoms, pick carbons
        for idx, sym in enumerate(symbols):
            if sym != 'C':
                continue
            # Get neighbor symbols
            nbr_syms = [symbols[j] for j in adjacency[idx]]
            # Determine bond type
            bondtype = self.get_bondtype(nbr_syms)
            classification[idx] = bondtype

        result = {}
        for idx, bondtype in classification.items():
            if result.get(bondtype) is None:
                result[bondtype] = []
            result[bondtype].append(idx)
        result = {k: len(v) for k, v in result.items()}
        return result

    def build_adjacency(self):
        """
        Build adjacency list based on cutoff distances with periodic boundary conditions.
        Returns:
            adjacency: list of sets, adjacency[i] is the set of neighbor indices of atom i
        """
        # Ensure wrapped within cell
        self.atoms.wrap()
        positions = self.atoms.get_positions()
        # Clean tiny negative values
        positions[positions < -1e-10] = 0.0
        # PBC box lengths
        cell_lengths = self.atoms.get_cell().lengths()
        # Build k-d tree
        tree = cKDTree(positions, boxsize=cell_lengths)
        max_cutoff = np.max(self.cutoff_matrix)
        # Sparse distance in COO
        pairs = tree.sparse_distance_matrix(tree, max_cutoff, output_type='coo_matrix')
        num_atoms = len(self.atoms)
        adjacency = [set() for _ in range(num_atoms)]
        row, col, dist = pairs.row, pairs.col, pairs.data
        atom_types = np.array([self.symbol_to_index[s] for s in self.atoms.get_chemical_symbols()])
        # Filter by element-specific cutoff
        valid = dist <= self.cutoff_matrix[atom_types[row], atom_types[col]]
        for i, j in zip(row[valid], col[valid]):
            if i != j:
                adjacency[i].add(j)
                adjacency[j].add(i)
        return adjacency

    @staticmethod
    def get_bondtype(symbols):
        """
        Classify based on the multiset of neighbor symbols.
        Input:
            symbols: list or space-separated string of neighbor element symbols
        Returns:
            classification string
        """
        case_dict = {
            ('C',): 'CX',
            ('C', 'H'): 'CX',
            ('Si',): 'SiC_cluster',
            ('H', 'Si'): 'SiC_cluster',
            ('C', 'Si'): 'SiC_cluster',
            ('C', 'H', 'Si'): 'SiC_cluster',
            ('F', 'Si'): 'SiC_cluster',
            ('F', 'H', 'Si'): 'SiC_cluster',
            ('C', 'F', 'Si'): 'SiC_cluster',
            ('C', 'F', 'H', 'Si'): 'SiC_cluster',
            ('C', 'O', 'Si'): 'SiC_cluster',
            ('C', 'H', 'O', 'Si'): 'SiC_cluster',
            ('C', 'F'): 'Fluorocarbon',
            ('C', 'F', 'H'): 'Fluorocarbon',
            ('C', 'O'): 'Fluorocarbon',
            ('C', 'H', 'O'): 'Fluorocarbon',
            ('C', 'F', 'O'): 'Fluorocarbon',
            ('C', 'F', 'H', 'O'): 'Fluorocarbon',
            ('C', 'F', 'O', 'Si'): 'Fluorocarbon',
            ('C', 'F', 'H', 'O', 'Si'): 'Fluorocarbon',
        }
        if symbols is None:
            return 'etc'
        if isinstance(symbols, str):
            symbols = symbols.split()
        key = tuple(sorted(set(symbols)))
        out = case_dict.get(key)
        if out == 'CX':
            # Count number of C bonds
            count_C = symbols.count('C')
            return f"C{count_C}"
        return out if out is not None else 'etc'

class DensityProfilePlotter:
    def run(self, profile, result, path_output):
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})
        fig, axes = plt.subplots(1, 3, figsize=(3.5 * 1.5, 7.1))
        ax_left, ax_middle, ax_right = axes
        self.draw_atomwise_density(ax_left, profile)
        self.draw_ratio(ax_middle, profile)
        self.draw_normalized_ratio(ax_right, profile)

        self.draw_region_boundary(ax_left, result, fill_regions=False)
        self.draw_region_boundary(ax_middle, result, fill_regions=True)
        self.draw_region_boundary(ax_right, result, fill_regions=True)

        self.add_text(ax_left, '(b)')
        self.add_text(ax_middle, '(c)')
        self.add_text(ax_right, '(d)')

        fig.tight_layout()
        fig.savefig(path_output, dpi=200)

    def draw_atomwise_density(self, ax, profile):
        for elem in PARAMS.ELEM_LIST:
            if elem not in profile:
                continue
            ax.plot(profile[elem], profile['z'], label=elem,
                    color=PARAMS.ATOM_COLOR[elem], zorder=1)
        ax.set_xlabel('Atom density')
        ax.set_ylabel(r'Height ($\mathrm{\AA}$)')
        ax.set_xlim(0, None)
        ax.set_ylim(0, 200)
        ax.legend()

    def draw_ratio(self, ax, profile):
        # prof_CHF = profile['C'] + profile['H'] + profile['F']
        prof_CHF = profile['C'] + profile['F']
        prof_all = profile['Si'] + profile['O'] + prof_CHF
        prof_ratio = prof_CHF / (prof_all + 1e-8)
        ax.plot(prof_ratio, profile['z'], label='CHF/all', color='black', zorder=1)
        ax.set_xlabel('CHF/SiOCHF')
        ax.yaxis.label.set_visible(False)
        ax.yaxis.set_ticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 200)
        ax.axvline(0.1, color='gray', linestyle='--', zorder=0, label='Mixed Layer Cutoff')
        ax.axvline(0.6, color='gray', linestyle='--', zorder=0, label='Film Layer Cutoff')

    def draw_normalized_ratio(self, ax, profile):
        prof_CHF = profile['C'] + profile['F']
        normalized_ratio = prof_CHF / np.max(prof_CHF)
        ax.plot(normalized_ratio, profile['z'], label='CHF/CHF_max', color='red', zorder=1)
        ax.set_xlabel('CHF/max(CHF)')
        ax.yaxis.label.set_visible(False)
        ax.yaxis.set_ticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 200)
        ax.axvline(0.6, color='gray', linestyle='--', zorder=0, label='Film Layer Cutoff')

    def draw_region_boundary(self, ax, result, fill_regions=False):
        z_mixed_min = result['z_mixed_min']
        z_mixed_max = result['z_mixed_max']
        z_film_max = result['z_film_max']
        z_film_max = result['z_film_max']
        if z_mixed_min is not None:
            ax.axhline(z_mixed_min, color='gray', linestyle='--', zorder=0)
        if z_mixed_max is not None:
            ax.axhline(z_mixed_max, color='gray', linestyle='--', zorder=0)
        if z_film_max is not None:
            ax.axhline(z_film_max, color='gray', linestyle='--', zorder=0)
        if fill_regions:
            ax.fill_between(x=ax.get_xlim(), y1=z_mixed_min, y2=z_mixed_max,
                            color='#d1ae4d', alpha=0.5, label='Mixed Layer',
                            zorder=0)
            ax.fill_between(x=ax.get_xlim(), y1=z_mixed_max, y2=z_film_max,
                            color='#3d2e04', alpha=0.5, label='Film Layer', zorder=0)

    def add_text(self, ax, text):
        ax.text(-0.25, 1.05, text, transform=ax.transAxes, ha='left', va='top',
                fontsize=10)

def main():
    image_list = {
            # 'density_profile_1.png': "/data_etch/data_HM/nurion/set_1/CF_100_coo/CF/100eV/rm_byproduct_str_shoot_20.coo",
            # 'density_profile_2.png': "/data_etch/data_HM/nurion/set_1/CF_100_coo/CF/100eV/rm_byproduct_str_shoot_1000.coo",
            # 'density_profile_2.png': "/data_etch/data_HM/nurion/set_3/CF_300_coo/CF/300eV/rm_byproduct_str_shoot_130.coo",
            # 'density_profile_3.png': "/data_etch/data_HM/nurion/set_1/CF_100_coo/CF/100eV/rm_byproduct_str_shoot_2000.coo",
            'density_profile_4.png': "/data_etch/data_HM/nurion/set_1/CF_100_coo/CF/100eV/rm_byproduct_str_shoot_4500.coo",
            # 'density_profile_5.png': "/data_etch/data_HM/nurion/set_1/CF_100_coo/CF/100eV/rm_byproduct_str_shoot_6690.coo",
            # 'density_profile_5.png': "/data_etch/data_HM/nurion/set_1/CF_100_coo/CF/100eV/rm_byproduct_str_shoot_9000.coo",
            # 'density_profile_5.png': "/data_etch/data_HM/nurion/set_2/CF_250_coo/CF/250eV/rm_byproduct_str_shoot_830.coo",
            }

    dpg = DensityProfileGenerator()
    dpa = DensityProfileAnalyzer()
    dpl = DensityProfilePlotter()

    for path_output, path_image in image_list.items():
        atoms = read(path_image, **PARAMS.LAMMPS_READ_OPTS)
        profile = dpg.run(atoms)
        analyze_result = dpa.run(atoms, profile)
        dpl.run(profile, analyze_result, path_output)


if __name__ == "__main__":
    main()
