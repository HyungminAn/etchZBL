from dataclasses import dataclass

from ase.io import read
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
    def run(self, profile):
        ct = CalculatorThickness()
        data = ct.run(profile, PARAMS.CUTOFF_RATIO_FILM, PARAMS.CUTOFF_RATIO_MIXED)
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

class DensityProfilePlotter:
    def run(self, axes_config, ad_rbd, path_output):
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})
        n_axis = len(axes_config)
        fig, axes = plt.subplots(1, n_axis, figsize=(3.5 * 1.5, 7.1))
        if n_axis == 1:
            axes = [axes]

        for ax, (text, (ad, fill_regions)) in zip(axes, axes_config.items()):
            ad.run(ax)
            ad_rbd.run(ax, fill_regions=fill_regions)
            self.add_text(ax, text)

        fig.tight_layout()
        fig.savefig(f'{path_output}.png', dpi=200)
        fig.savefig(f'{path_output}.eps')
        fig.savefig(f'{path_output}.pdf')

    def add_text(self, ax, text):
        ax.text(-0.25, 1.05, text,
                transform=ax.transAxes, ha='left', va='top', fontsize=10)

class AxisDrawerAtomwiseDensity:
    def __init__(self, profile):
        self.profile = profile

    def run(self, ax):
        profile = self.profile
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

class AxisDrawerRatio:
    def __init__(self, profile):
        self.profile = profile

    def run(self, ax):
        profile = self.profile
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

class AxisDrawerNormalizedRatio:
    def __init__(self, profile):
        self.profile = profile

    def run(self, ax):
        profile = self.profile
        prof_CHF = profile['C'] + profile['F']
        normalized_ratio = prof_CHF / np.max(prof_CHF)
        ax.plot(normalized_ratio, profile['z'], label='CHF/CHF_max', color='red', zorder=1)
        ax.set_xlabel('CHF/max(CHF)')
        ax.yaxis.label.set_visible(False)
        ax.yaxis.set_ticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 200)
        ax.axvline(0.6, color='gray', linestyle='--', zorder=0, label='Film Layer Cutoff')

class AxisDrawerRegionBoundary:
    def __init__(self, result):
        self.result = result

    def run(self, ax, fill_regions=False):
        result = self.result
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

class CalculatorThickness:
    def run(self, profile, cutoff_ratio_film, cutoff_ratio_mixed):
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

class CalculatorBondDensity:
    def run(self, atoms, z_min, z_max, bin_width=0.5):
        """
        Compute bond density profiles for specified bond types within a z-range.

        The z-range is divided into bins of `bin_width`. The density is
        calculated as the number of bonds per bin divided by the bin volume.

        Args:
            atoms (ase.Atoms): An Atoms object containing positions and symbols.
            z_min (float): The minimum z-coordinate to consider.
            z_max (float): The maximum z-coordinate to consider.
            bin_width (float): The width of the z-bins for density calculation.

        Returns:
            dict: A dictionary containing z-bin centers and density profiles for
                  each bond category. Returns None if no atoms are found or
                  if the z-range is invalid.
        """
        if z_min is None or z_max is None or z_min >= z_max:
            return None

        # Positions and cell dimensions for PBC
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()

        # Assumes orthorhombic cell: use diagonal entries
        box = atoms.get_cell().diagonal()
        if np.any(box <= 0):
            print("Error: Invalid or non-orthorhombic cell dimensions.")
            return None

        # Filter atoms by z-range
        z_coords = positions[:, 2]
        z_mask = (z_coords >= z_min) & (z_coords <= z_max)
        if not np.any(z_mask):
            return None

        # Map element to cutoff-matrix index.
        # This mapping must be consistent with the provided cutoff_matrix.
        # For this example, we'll use a hardcoded map and dummy data.
        symbol_map = {'Si': 0, 'O': 1, 'C': 2, 'H': 3, 'F': 4}

        # Define the bond categories based on element groups
        group1_elements = {'Si', 'O'}
        group2_elements = {'C', 'H', 'F'}

        # Load pairwise cutoff distances.
        cutoff_mat = np.loadtxt('cutoff_matrix.npy')

        # Build the cKDTree for all atoms in the system
        tree = cKDTree(positions, boxsize=box)

        # Use a max cutoff for the initial neighbor search
        max_cut = cutoff_mat.max()

        # Query all pairs within max_cut. sparse_distance_matrix returns a dict
        # {(i, j): distance} for pairs i, j.
        dist_dict = tree.sparse_distance_matrix(tree, max_cut)

        # Convert to arrays for vectorized filtering
        pairs = np.array(list(dist_dict.keys()), dtype=int)
        dists = np.array(list(dist_dict.values()))

        # Separate indices of the pair
        i_global, j_global = pairs[:, 0], pairs[:, 1]

        # Remove self-pairs (i == j)
        self_mask = (i_global == j_global)
        i_global = i_global[~self_mask]
        j_global = j_global[~self_mask]
        dists = dists[~self_mask]

        # Use only unique pairs (i < j) to avoid double counting bonds
        # Note: cKDTree returns both (i, j) and (j, i)
        unique_pair_mask = (i_global < j_global)
        i_global = i_global[unique_pair_mask]
        j_global = j_global[unique_pair_mask]
        dists = dists[unique_pair_mask]

        # Vectorized check for valid bonds based on pairwise cutoffs
        i_types = np.array([symbol_map.get(symbols[i], -1) for i in i_global])
        j_types = np.array([symbol_map.get(symbols[j], -1) for j in j_global])

        # Check for invalid symbols (not in symbol_map)
        valid_symbol_mask = (i_types != -1) & (j_types != -1)
        i_global = i_global[valid_symbol_mask]
        j_global = j_global[valid_symbol_mask]
        dists = dists[valid_symbol_mask]
        i_types = i_types[valid_symbol_mask]
        j_types = j_types[valid_symbol_mask]

        # Get the correct cutoff for each valid pair
        valid_cutoffs = cutoff_mat[i_types, j_types]

        # Filter for bonds within their specific cutoff
        valid_bond_mask = dists <= valid_cutoffs

        # Get the final list of valid bonds
        bond_i_idx = i_global[valid_bond_mask]
        bond_j_idx = j_global[valid_bond_mask]

        # Calculate the z-coordinate of the midpoint for each bond
        bond_z_midpoints = (positions[bond_i_idx, 2] + positions[bond_j_idx, 2]) / 2.0

        # Filter bonds to only those with midpoints in the specified z-range
        midpoint_mask = (bond_z_midpoints >= z_min) & (bond_z_midpoints <= z_max)
        bond_i_idx = bond_i_idx[midpoint_mask]
        bond_j_idx = bond_j_idx[midpoint_mask]
        bond_z_midpoints = bond_z_midpoints[midpoint_mask]

        # Initialize bins for bond counts
        num_bins = int(np.ceil((z_max - z_min) / bin_width))
        bins = np.linspace(z_min, z_max, num_bins + 1)
        bin_indices = np.digitize(bond_z_midpoints, bins) - 1 # -1 to make it 0-indexed

        # Initialize count arrays for each bond category
        counts_cat1 = np.zeros(num_bins, dtype=int)
        counts_cat2 = np.zeros(num_bins, dtype=int)
        counts_cat3 = np.zeros(num_bins, dtype=int)

        # Categorize and count bonds
        for i, j, bin_idx in zip(bond_i_idx, bond_j_idx, bin_indices):
            if bin_idx >= num_bins or bin_idx < 0:
                continue

            sym_i, sym_j = symbols[i], symbols[j]
            is_i_g1 = sym_i in group1_elements
            is_j_g1 = sym_j in group1_elements
            is_i_g2 = sym_i in group2_elements
            is_j_g2 = sym_j in group2_elements

            if is_i_g1 and is_j_g1:
                counts_cat1[bin_idx] += 1
            elif is_i_g2 and is_j_g2:
                counts_cat2[bin_idx] += 1
            else:
                # All other bonds (Si-C, C-O, etc.)
                counts_cat3[bin_idx] += 1

        # Calculate bin volume
        bin_volume = box[0] * box[1] * bin_width

        # Calculate densities (bonds per volume)
        density_cat1 = counts_cat1 / bin_volume
        density_cat2 = counts_cat2 / bin_volume
        density_cat3 = counts_cat3 / bin_volume

        # Get the center of each bin for plotting
        z_bin_centers = (bins[:-1] + bins[1:]) / 2

        # Return the results
        return {
            'z_bins': z_bin_centers,
            'Si_O_bonds': density_cat1,
            'C_H_F_bonds': density_cat2,
            'Other_bonds': density_cat3,
        }

class AxisDrawerBondDensity:
    def __init__(self, bond_profile):
        self.bond_profile = bond_profile

    def run(self, ax):
        x = self.bond_profile['z_bins']
        y1 = self.bond_profile['Si_O_bonds']
        y2 = self.bond_profile['C_H_F_bonds']
        y3 = self.bond_profile['Other_bonds']
        ax.plot(y1, x, label='Si-O', color='blue', zorder=1)
        ax.plot(y2, x, label='C-H-F', color='green', zorder=1)
        ax.plot(y3, x, label='others', color='red', zorder=1)

        ax.set_xlim(0, None)
        ax.set_ylim(0, 200)
        ax.set_xlabel('Bond Density (bonds/Å³)')
        ax.set_ylabel(r'Height ($\mathrm{\AA}$)')
        ax.legend(loc='upper right', fontsize=10, frameon=False)

def main():
    image_list = {
            # 'result.png': "/data_etch/data_HM/nurion/set_1/CF_100_coo/CF/100eV/rm_byproduct_str_shoot_4500.coo",
            'result': "str.coo",
            }

    dpg = DensityProfileGenerator()
    dpa = DensityProfileAnalyzer()
    dpl = DensityProfilePlotter()
    # dpb = CalculatorBondDensity()

    for path_output, path_image in image_list.items():
        atoms = read(path_image, **PARAMS.LAMMPS_READ_OPTS)
        profile = dpg.run(atoms)
        analyze_result = dpa.run(profile)
        # bond_result = dpb.run(atoms, 0, 200)

        axes_config = {
            '(b)': (AxisDrawerAtomwiseDensity(profile), False),
            '(c)': (AxisDrawerRatio(profile), True),
            '(d)': (AxisDrawerNormalizedRatio(profile), True),
            # '(c)': (AxisDrawerBondDensity(bond_result), False),
        }
        ad_rbd = AxisDrawerRegionBoundary(analyze_result)
        dpl.run(axes_config, ad_rbd, path_output)

if __name__ == "__main__":
    main()
