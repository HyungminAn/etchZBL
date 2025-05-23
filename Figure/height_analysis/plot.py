import os
import sys
from functools import wraps
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

from ase.io import read
from ase.data import atomic_numbers, atomic_masses


@dataclass
class PARAMS:
    @dataclass
    class LAMMPS:
        ATOM_IDX_Si, ATOM_NUM_Si = 1, 14
        ATOM_IDX_O, ATOM_NUM_O = 2, 8
        ATOM_IDX_C, ATOM_NUM_C = 3, 6
        ATOM_IDX_H, ATOM_NUM_H = 4, 1
        ATOM_IDX_F, ATOM_NUM_F = 5, 9

        READ_OPTS = {
            'format': 'lammps-data',
            'Z_of_type': {
                ATOM_IDX_Si: ATOM_NUM_Si,
                ATOM_IDX_O: ATOM_NUM_O,
                ATOM_IDX_C: ATOM_NUM_C,
                ATOM_IDX_H: ATOM_NUM_H,
                ATOM_IDX_F: ATOM_NUM_F,
            }
        }

    @dataclass
    class NUMPY:
        SAVE_OPTS = {
            'header': 'key height(A)',
            'comments': ''
        }

    @dataclass
    class PLOT:
        @dataclass
        class COLORS:
            COLOR_LIST = {
                '1': '#075c29',
                '2': '#609af7',
                '3': '#de3535',
                'layer_mixed': '#d1ae4d',
                'layer_film': '#3d2e04',
                'default': '#609af7',
            }

            COLORS = {
                'CF_500': COLOR_LIST['1'],
                'CF_750': COLOR_LIST['1'],
                'CF_1000': COLOR_LIST['1'],

                'CF2_75': COLOR_LIST['1'],
                'CF2_100': COLOR_LIST['1'],
                'CF2_250': COLOR_LIST['1'],
                'CF2_500': COLOR_LIST['1'],
                'CF2_750': COLOR_LIST['1'],
                'CF2_1000': COLOR_LIST['1'],

                'CF3_25': COLOR_LIST['1'],
                'CF3_50': COLOR_LIST['1'],
                'CF3_100': COLOR_LIST['1'],
                'CF3_250': COLOR_LIST['1'],
                'CF3_500': COLOR_LIST['1'],
                'CF3_750': COLOR_LIST['1'],
                'CF3_1000': COLOR_LIST['1'],

                'CH2F_750': COLOR_LIST['1'],
                'CH2F_1000': COLOR_LIST['1'],

                'CHF2_250': COLOR_LIST['1'],
                'CHF2_500': COLOR_LIST['1'],
                'CHF2_750': COLOR_LIST['1'],
                'CHF2_1000': COLOR_LIST['1'],

                'CF3_10': COLOR_LIST['3'],
                'CF2_25': COLOR_LIST['3'],
                'CH2F_100': COLOR_LIST['3'],
                }

        @dataclass
        class HEIGHT:
            READ_INTERVAL = 10
            CUTOFF_PERCENTILE = 98  # percentile
            SHIFT = 6.0
            CARBON_FILM_CUTOFF = (85, 15)  # percentile
            TRUNCATE_INITIAL_REGION = 0.2  # 10^16 cm-2

        @dataclass
        class ATOMDENSITY:
            SPACING = 0.1  # density profile
            BW_WIDTH = 0.2  # density profile
            CUTOFF_RATIO_FILM = 0.6  # density profile
            CUTOFF_RATIO_MIXED = 0.1  # density profile
            CUTOFF_RATIO_FILM_UPPER = 0.5  # density profile
            ELEM_LIST = ['Si', 'O', 'C', 'H', 'F']

    @dataclass
    class CONVERT:
        ANGST_TO_NM = 0.1
        CONV_FACTOR_TO_CM2 = 1/9000  # 9000 incidence corresponds to 10^17 cm^2
        ION_CONVERT_DICT = {
                'CF': 'CF$^{+}$',
                'CF2': 'CF${}_{2}^{+}$',
                'CF3': 'CF${}_{3}^{+}$',
                'CH2F': 'CH${}_{2}$F$^{+}$',
                'CHF2': 'CHF${}_{2}^{+}$',
                }


class pklSaver:
    @staticmethod
    def run(func_gen_name):
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
                    y = data[:, 1:]
                    return x, y
                x, y = func(self, *args, **kwargs)
                np.savetxt(path_save, np.c_[x, y], **PARAMS.NUMPY.SAVE_OPTS)
                return x, y
            return wrapper
        return decorator

class ImageLoader:
    def run(self, src_list):
        PATTERN = 'rm_byproduct_str_shoot_'
        file_dict = self.get_file_list(src_list, PATTERN)
        keys = sorted(file_dict.keys())
        result = {}
        for key in keys[::PARAMS.PLOT.HEIGHT.READ_INTERVAL]:
            file = file_dict[key]
            atoms = read(file, **PARAMS.LAMMPS.READ_OPTS)
            result[key] = atoms
        return result

    @staticmethod
    def get_file_list(src_list, pattern):
        '''
        Get the file list from the source directories,
        starting with the given pattern.
        '''
        file_dict = {}
        for src in src_list:
            for file in os.listdir(src):
                if not file.startswith(pattern):
                    continue
                key = int(file.split('_')[-1].split('.')[0])
                if file_dict.get(key) is not None:
                    print(f'key {key} already exists')
                    continue
                file_dict[key] = os.path.join(src, file)
        return file_dict

class HeightChangeProcessor:
    def __init__(self, name):
        self.name = name

    @pklSaver.run(lambda self: f'{self.name}_shifted_height.txt')
    def run(self, images, src_list):
        x, y = [], []
        for key, image in images.items():
            height = np.percentile(image.get_positions()[:, 2],
                                   PARAMS.PLOT.HEIGHT.CUTOFF_PERCENTILE)
            print(f'{key}: {height} A')
            x.append(key)
            y.append(height)
        imgldr = ImageLoader()
        add_dict = imgldr.get_file_list(src_list, pattern='add_str_shoot_')
        sub_dict = imgldr.get_file_list(src_list, pattern='sub_str_shoot_')

        for key in add_dict.keys():
            y[x > key] -= PARAMS.PLOT.HEIGHT.SHIFT
        for key in sub_dict.keys():
            y[x > key] += PARAMS.PLOT.HEIGHT.SHIFT
        return x, y

class CarbonChangeProcessor:
    def __init__(self, name):
        self.name = name

    @pklSaver.run(lambda self: f'{self.name}_carbonfilm.txt')
    def run(self, images):
        x, y = [], []
        for key, image in images.items():
            t_carbon = self.calculate(key, image)
            x.append(key)
            y.append(t_carbon)
        x, y = np.array(x), np.array(y)
        return x, y

    def calculate(self, key, image):
        cut_upper, cut_lower = PARAMS.PLOT.HEIGHT.CARBON_FILM_CUTOFF
        pos = image.get_positions()[:, 2]
        symbols = image.get_chemical_symbols()
        idx_C = np.array([i for i, s in enumerate(symbols) if s == 'C'])
        if len(idx_C) > 0:
            z_max = np.percentile(pos[idx_C], cut_upper)
            z_min = np.percentile(pos[idx_C], cut_lower)
            t_carbon = z_max - z_min
        else:
            z_min = z_max = t_carbon = 0
        print(f'{key}: {t_carbon} ({z_max} - {z_min}) A')
        return z_min, z_max, t_carbon

class FilmAnalyzer:
    def __init__(self, name):
        self.name = name

    def run(self, images):
        path_save = f'{self.name}_thickness.txt'
        if os.path.exists(path_save):
            print(f"{path_save} already exists, loading data from it.")
            data = np.loadtxt(path_save, skiprows=1)
            x = data[:, 0].astype(float)
            y = data[:, 1:]
            return x, y

        x, y = [], []
        for key, image in images.items():
            result = self.run_single(image)
            x.append(key)
            value = [
                    result['h_mixed'],
                    result['h_film'],
                    result['density_mixed_layer'],
                    result['density_film_layer'],
                    result['fc_ratio_mixed_layer'],
                    result['fc_ratio_film_layer'],
                    ]
            value += [
                    result['spx_ratio_mixed_layer']['sp3'],
                    result['spx_ratio_mixed_layer']['sp2'],
                    result['spx_ratio_mixed_layer']['sp'],
                    result['spx_ratio_mixed_layer']['others'],
                    result['spx_ratio_film_layer']['sp3'],
                    result['spx_ratio_film_layer']['sp2'],
                    result['spx_ratio_film_layer']['sp'],
                    result['spx_ratio_film_layer']['others'],
                    ]
            y.append(value)
            line = (f"{key}: {result['h_mixed']} A, "
                    f"{result['h_film']} A, "
                    f"{result['density_mixed_layer']} g/cm^3, "
                    f"{result['density_film_layer']} g/cm^3, "
                    f"{result['fc_ratio_mixed_layer']}, "
                    f"{result['fc_ratio_film_layer']}"
                    f"{result['spx_ratio_mixed_layer']}, "
                    f"{result['spx_ratio_film_layer']}"
                    )
            print(line)
        x, y = np.array(x, dtype=float), np.array(y, dtype=float)
        mat = np.hstack((x.reshape(-1, 1), y))
        SAVE_OPTS = {
            'header': 'key h_mixed(A) h_film(A) ',
            'comments': '',
                }
        np.savetxt(path_save, mat, **SAVE_OPTS)
        return x, y

    def run_single(self, atoms):
        profile = self.get_density_profile(atoms)
        data = self.get_thickness(profile,
                                  PARAMS.PLOT.ATOMDENSITY.CUTOFF_RATIO_FILM,
                                  PARAMS.PLOT.ATOMDENSITY.CUTOFF_RATIO_MIXED,
                                  )
        density_mixed_layer = self.calculate_average_density(
                atoms, data['z_mixed_min'], data['z_mixed_max'])
        density_film_layer = self.calculate_average_density(
                atoms, data['z_film_min'], data['z_film_max'])
        fc_ratio_mixed_layer = self.calculate_FC_ratio(
                atoms, data['z_mixed_min'], data['z_mixed_max'])
        fc_ratio_film_layer = self.calculate_FC_ratio(
                atoms, data['z_film_min'], data['z_film_max'])
        spx_ratio_mixed_layer = self.calculate_spx_ratio(
                atoms, data['z_mixed_min'], data['z_mixed_max'])
        spx_ratio_film_layer = self.calculate_spx_ratio(
                atoms, data['z_film_min'], data['z_film_max'])

        data.update({
            'density_mixed_layer': density_mixed_layer,
            'density_film_layer': density_film_layer,
            'fc_ratio_mixed_layer': fc_ratio_mixed_layer,
            'fc_ratio_film_layer': fc_ratio_film_layer,
            'spx_ratio_mixed_layer': spx_ratio_mixed_layer,
            'spx_ratio_film_layer': spx_ratio_film_layer,
        })

        return data

    def get_density_profile(self, atoms):
        pos_z = atoms.get_positions()[:, 2]
        symbols = atoms.get_chemical_symbols()

        x = np.arange(0, atoms.get_cell()[2, 2], PARAMS.PLOT.ATOMDENSITY.SPACING)
        result = {}
        result['z'] = x
        for elem in PARAMS.PLOT.ATOMDENSITY.ELEM_LIST:
            mask = np.array([idx for idx, s in enumerate(symbols) if s == elem])
            if len(mask) <= 1:
                continue
            y = pos_z[mask]
            gaussian = gaussian_kde(y, bw_method=PARAMS.PLOT.ATOMDENSITY.BW_WIDTH)
            density = gaussian.evaluate(x) * len(y)
            result[elem] = density
        return result

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
        fc_ratio = len(mask_F) / len(mask_C)
        return fc_ratio

    def calculate_spx_ratio(self, atoms, z_min, z_max):
        """
        Compute spx hybridization counts for carbon atoms within a specified z-range,
        accounting for periodic boundary conditions and per-pair cutoffs.

        Returns:
            dict: {'sp3': int, 'sp2': int, 'sp': int, 'others': int}
        """
        result = {'sp3': 0, 'sp2': 0, 'sp': 0, 'others': 0}
        if z_min is None or z_max is None:
            return result

        # Positions and cell dimensions for PBC
        positions = atoms.get_positions()
        # Assumes orthorhombic cell: use diagonal entries
        box = atoms.get_cell().diagonal()
        tree = cKDTree(positions, boxsize=box)

        # Filter atoms by z-range
        z_coords = positions[:, 2]
        z_mask = (z_coords >= z_min) & (z_coords <= z_max)
        if not np.any(z_mask):
            return result

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
            return result

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
        result = {
            'sp3': int(np.count_nonzero(counts == 4)),
            'sp2': int(np.count_nonzero(counts == 3)),
            'sp': int(np.count_nonzero(counts == 2)),
            'others': int(np.count_nonzero(~np.isin(counts, [2, 3, 4])))
        }

        # Debug output
        debug = False
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

        return result

class DataProcessor:
    def __init__(self, name):
        self.name = name

    def run(self, src_list):
        images = ImageLoader().run(src_list)

        x_mod, y_mod = HeightChangeProcessor(self.name).run(images, src_list)
        x_carbon, y_carbon = CarbonChangeProcessor(self.name).run(images)
        x_layer, y_layer = FilmAnalyzer(self.name).run(images)

        result = {
                'height_change': (x_mod, y_mod),
                'carbon_thickness': (x_carbon, y_carbon),
                'mixed_thickness': (x_layer, y_layer),
                }
        return result

class DataPlotter:
    def run(self, data):
        fig, ax_dict = self.generate_figure(data)

        for system, (ax, ax_carbon) in ax_dict.items():
            self.normalize_data(data[system])
            self.plot_height(data, system, ax)
            self.plot_carbon(data, system, ax_carbon)
            self.decorate_axes((ax, ax_carbon), system)

        self.set_ylim(ax_dict)
        self.save_figure(fig)

    def generate_figure(self, data):
        '''
        Generate the figure and axes for the plot.
        '''
        plt.rcParams.update({'font.family': 'Arial'})

        set_ion, set_energy = set(), set()
        for system in data.keys():
            ion, energy = system.split('_')
            set_ion.add(ion)
            set_energy.add(energy)
        set_ion = sorted(list(set_ion))
        set_energy = sorted(list(set_energy), key=lambda x: int(x))
        n_ion = len(set_ion)
        n_energy = len(set_energy)

        n_row, n_col = n_ion, n_energy
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3),)
        ax_dict = {}
        for system in data.keys():
            ion, energy = system.split('_')
            ax_dict[system] = axes[set_ion.index(ion), set_energy.index(energy)]
        for ion in set_ion:
            for energy in set_energy:
                key = f'{ion}_{energy}'
                if key not in ax_dict:
                    fig.delaxes(axes[set_ion.index(ion), set_energy.index(energy)])

        ax_dict_new = {}
        for key, ax in ax_dict.items():
            ax_dict_new[key] = (ax, ax.twinx())
        ax_dict = ax_dict_new

        return fig, ax_dict

    def plot_height(self, data, system, ax):
        '''
        Plot the height change.
        '''
        x_mod, y_mod = data[system]['height_change']
        color = PARAMS.PLOT.COLORS.COLORS.get(system, PARAMS.PLOT.COLORS.COLOR_LIST['default'])
        ax.plot(x_mod, y_mod, 'o-', markersize=2, color=color, alpha=0.5)
        print(f'{system}: Etched thickness {np.min(y_mod)}')

    def plot_carbon(self, data, system, ax_carbon):
        '''
        Plot the carbon film thickness.
        '''
        x_carbon_front, y_carbon_front, x_carbon_back, y_carbon_back = \
            data[system]['carbon_thickness']
        ax_carbon.plot(x_carbon_front, y_carbon_front,
                       '--', markersize=2, color='black', alpha=0.2)
        ax_carbon.plot(x_carbon_back, y_carbon_back,
                       '-', markersize=2, color='black', alpha=0.7)

    def normalize_data(self, data):
        '''
        Normalize the data to the range [0, 1].
        '''
        self.normalize_height(data)
        self.normalize_carbon(data)

    def normalize_height(self, data):
        '''
        Normalize the height data to the range [0, 1].
        '''
        x_mod, y_mod = data['height_change']
        x_mod *= PARAMS.CONVERT.CONV_FACTOR_TO_CM2
        y_mod *= PARAMS.CONVERT.ANGST_TO_NM
        y_mod -= y_mod[0]
        data['height_change'] = (x_mod, y_mod)

    def normalize_carbon(self, data):
        x_carbon, y_carbon = data['carbon_thickness']
        x_carbon *= PARAMS.CONVERT.CONV_FACTOR_TO_CM2
        y_carbon *= PARAMS.CONVERT.ANGST_TO_NM
        x_carbon_front = x_carbon[x_carbon < PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]
        y_carbon_front = y_carbon[x_carbon < PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]
        x_carbon_back = x_carbon[x_carbon >= PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]
        y_carbon_back = y_carbon[x_carbon >= PARAMS.PLOT.HEIGHT.TRUNCATE_INITIAL_REGION]
        data['carbon_thickness'] = (x_carbon_front, y_carbon_front,
                                    x_carbon_back, y_carbon_back)

    def decorate_axes(self, axes, system):
        '''
        Decorate the axes with titles and labels.
        '''
        ax, ax_carbon = axes
        self.decorate_axes_height(ax, system)
        self.decorate_axes_carbon(ax_carbon)

    def decorate_axes_height(self, ax, system):
        ion, energy = system.split('_')
        title = f'{PARAMS.CONVERT.ION_CONVERT_DICT[ion]}, {energy} eV'
        ax.set_title(title)
        ax_color = PARAMS.PLOT.COLORS.COLORS.get(system, PARAMS.PLOT.COLORS.COLOR_LIST['default'])
        ax.set_xlabel('Ion dose (10$^{17}$ cm$^{-2}$)')
        ax.set_ylabel('Height change (nm)', color=ax_color)
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlim(0, 1.0)
        ax.tick_params(axis='y', colors=ax_color)

    def decorate_axes_carbon(self, ax_carbon):
        ax_carbon.set_ylabel('Carbon film thickness (nm)')
        ax_carbon.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax_carbon.set_xlim(0, 1.0)

    def set_ylim(self, ax_dict):
        '''
        Set the y-limits of the axes to be the same for all systems.
        '''
        y_min, y_max = 0, 0
        for system, (ax, ax_carbon) in ax_dict.items():
            y_min_1, y_max_1 = ax.get_ylim()
            if y_min_1 < y_min:
                y_min = y_min_1
            if y_max_1 > y_max:
                y_max = y_max_1
            y_min_2, y_max_2 = ax_carbon.get_ylim()
            if y_min_2 < y_min:
                y_min = y_min_2
            if y_max_2 > y_max:
                y_max = y_max_2
        # y_min = -20

        for system, (ax, ax_carbon) in ax_dict.items():
            ax.set_ylim(y_min, y_max)
            ax_carbon.set_ylim(y_min, y_max)

    def save_figure(self, fig):
        '''
        Save the figure in different formats.
        '''
        fig.tight_layout()
        fig.savefig('result.png', dpi=200)
        fig.savefig('result.pdf')
        fig.savefig('result.eps')

class DataPlotterSelected(DataPlotter):
    def run(self, data):
        fig, ax_dict = self.generate_figure(data)

        for system, axes in ax_dict.items():
            (ax_change, ax_carbon), ax_mixed = axes
            self.normalize_data(data[system])
            self.plot_height(data, system, ax_change)
            self.plot_carbon(data, system, ax_carbon)
            self.plot_mixed(data, system, ax_mixed)
            self.decorate_axes(axes, system)

        self.set_ylim(ax_dict)
        self.save_figure(fig)

    def generate_figure(self, data):
        plt.rcParams.update({'font.family': 'Arial'})
        n_row, n_col = 5, len(data)
        fig, axes = plt.subplots(n_row, n_col, figsize=(3*n_col, 3*n_row),)
        ax_dict = {}
        for idx, system in enumerate(data.keys()):
            ax_dict[system] = axes[:, idx]

        ax_dict_new = {}
        n_twin = 1
        for key, axes in ax_dict.items():
            n_axes = len(axes)
            ax_dict_new[key] = []
            for i in range(n_twin):
                ax_dict_new[key].append((axes[i], axes[i].twinx()))
            for i in range(n_axes - n_twin):
                ax_dict_new[key].append(axes[n_twin + i])
        ax_dict = ax_dict_new

        return fig, ax_dict

    def normalize_data(self, data):
        '''
        Normalize the data to the range [0, 1].
        '''
        self.normalize_height(data)
        self.normalize_carbon(data)
        self.normalize_mixed(data)

    def normalize_mixed(self, data):
        '''
        Normalize the mixed data to the range [0, 1].
        '''
        x_layer, y_layer = data['mixed_thickness']
        x_layer *= PARAMS.CONVERT.CONV_FACTOR_TO_CM2
        y_layer *= PARAMS.CONVERT.ANGST_TO_NM
        y_layer -= y_layer[0]
        data['mixed_thickness'] = (x_layer, y_layer)

    def plot_mixed(self, data, system, ax):
        '''
        Plot the mixed layer thickness.
        '''
        x, y = data[system]['mixed_thickness']
        y_mixed, y_film = y[:, 0], y[:, 1]
        colors = [PARAMS.PLOT.COLORS.COLOR_LIST['layer_mixed'],
                 PARAMS.PLOT.COLORS.COLOR_LIST['layer_film']]
        ax.stackplot(x, y_mixed, y_film, colors=colors,)

    def decorate_axes(self, axes, system):
        '''
        Decorate the axes with titles and labels.
        '''
        (ax_height, ax_carbon), ax_mixed = axes
        self.decorate_axes_height(ax_height, system)
        self.decorate_axes_carbon(ax_carbon)
        self.decorate_axes_mixed(ax_mixed)

    def decorate_axes_mixed(self, ax):
        ax.set_ylabel('Mixed layer thickness (nm)')
        ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlim(0, 1.0)

    def set_ylim(self, ax_dict):
        '''
        Set the y-limits of the axes to be the same for all systems.
        '''
        y_min = np.min([min(ax.get_ylim()[0], ax_carbon.get_ylim()[0])
                        for ((ax, ax_carbon), _)in ax_dict.values()])
        y_max = np.max([max(ax.get_ylim()[1], ax_carbon.get_ylim()[1])
                        for ((ax, ax_carbon), _) in ax_dict.values()])
        for ((ax, ax_carbon), _) in ax_dict.values():
            ax.set_ylim(y_min, y_max)
            ax_carbon.set_ylim(y_min, y_max)

        y_min = np.min([ax_mixed.get_ylim()[0] for (_, ax_mixed) in ax_dict.values()])
        y_max = np.max([ax_mixed.get_ylim()[1] for (_, ax_mixed) in ax_dict.values()])
        for (_, ax_mixed) in ax_dict.values():
            ax_mixed.set_ylim(y_min, y_max)

def main():
    if len(sys.argv) != 2:
        print('Usage: python get_height.py input.yaml')
        sys.exit(1)

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        inputs = yaml.safe_load(f)
    data = {}
    for ion in inputs.keys():
        for energy, src_list in inputs[ion].items():
            key = f'{ion}_{energy}'
            dp = DataProcessor(key)
            print(f'{key}')
            data[key] = dp.run(src_list)
    # dplot = DataPlotter()
    # dplot.run(data)
    dplot = DataPlotterSelected()
    dplot.run(data)

if __name__ == '__main__':
    main()
