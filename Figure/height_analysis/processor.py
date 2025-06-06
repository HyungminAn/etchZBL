import os
from abc import ABC, abstractmethod
import pickle

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

from ase.data import atomic_numbers, atomic_masses

from imageloader import ImageLoader
from params import PARAMS
from params import pklSaver

class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self, name):
        self.name = name
        self.filename_suffix = None

    def check_exists(self):
        """
        Check if the processed data already exists.
        :return: True if exists, False otherwise
        """
        path_save = f'{self.name}_{self.filename_suffix}'
        print(f'Checking if {path_save} exists...: {os.path.exists(path_save)}')
        return os.path.exists(path_save)

class HeightChangeProcessor(BaseProcessor):
    def __init__(self, name, suffix):
        self.name = name
        self.filename_suffix = suffix

    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images, src_list):
        x, y = [], []
        for key, image in images.items():
            height = np.percentile(image.get_positions()[:, 2],
                                   PARAMS.PLOT.HEIGHT.CUTOFF_PERCENTILE)
            print(f'{key}: {height} A')
            x.append(key)
            y.append(height)
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        # imgldr = ImageLoader()
        # add_dict = imgldr.get_file_list(src_list, pattern='add_str_shoot_')
        # sub_dict = imgldr.get_file_list(src_list, pattern='sub_str_shoot_')

        # for key in add_dict.keys():
        #     y[x > key] -= PARAMS.PLOT.HEIGHT.SHIFT
        # for key in sub_dict.keys():
        #     y[x > key] += PARAMS.PLOT.HEIGHT.SHIFT

        labels = ['key', 'height(A)']
        return x, y, labels

class AverageDensityProcessor(BaseProcessor):
    def __init__(self, name, suffix):
        self.name = name
        self.filename_suffix = suffix

    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images, height_dict):
        x, y = [], []
        for key, image in images.items():
            z_min, z_max, _ = height_dict.get(key, (None, None, None))
            density = self.run_single(image, z_min, z_max)
            x.append(key)
            y.append(density)
            print(f'{key}: {density} g/cm^3')
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        labels = ['key', 'density(g/cm^3)']
        return x, y, labels

    def run_single(self, atoms, z_min, z_max):
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

class FCratioProcessor(BaseProcessor):
    def __init__(self, name, suffix):
        self.name = name
        self.filename_suffix = suffix

    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images, height_dict):
        x, y = [], []
        for key, image in images.items():
            z_min, z_max, _ = height_dict.get(key, (None, None, None))
            fc_ratio = self.run_single(image, z_min, z_max)
            x.append(key)
            y.append(fc_ratio)
            print(f'{key}: {fc_ratio} F/C ratio')
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        labels = ['key', 'FC_ratio']
        return x, y, labels

    def run_single(self, atoms, z_min, z_max):
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
        if len(mask_C) == 0 or len(mask_F) == 0:
            return None
        fc_ratio = len(mask_F) / len(mask_C)
        return fc_ratio

class ProfileProcessor(BaseProcessor):
    def __init__(self, name, suffix, system=None):
        self.system = system
        self.name = name
        self.filename_suffix = suffix
        if self.system == 'SiO2':
            self.elem_list = PARAMS.PLOT.ATOMDENSITY.ELEM_LIST_SiO2
        elif self.system == 'Si3N4':
            self.elem_list = PARAMS.PLOT.ATOMDENSITY.ELEM_LIST_Si3N4
        else:
            raise ValueError(f"Unknown system: {self.system}")

    def run(self, images):
        path_save = f'{self.name}_{self.filename_suffix}'
        if os.path.exists(path_save):
            print(f'Loading existing profile data from {path_save}')
            with open(path_save, 'rb') as f:
                result = pickle.load(f)
            return result

        result = {}
        for key, image in images.items():
            result[key] = self.run_single(image)
            print(f'Profile {key} Done')

        print(f'Saving profile data to {path_save}')
        with open(path_save, 'wb') as f:
            pickle.dump(result, f)
        return result

    def run_single(self, atoms):
        pos_z = atoms.get_positions()[:, 2]
        symbols = atoms.get_chemical_symbols()

        x = np.arange(0, atoms.get_cell()[2, 2], PARAMS.PLOT.ATOMDENSITY.SPACING)
        result = {}
        result['z'] = x
        for elem in self.elem_list:
            mask = np.array([idx for idx, s in enumerate(symbols) if s == elem])
            if len(mask) <= 1:
                continue
            y = pos_z[mask]
            gaussian = gaussian_kde(y, bw_method=PARAMS.PLOT.ATOMDENSITY.BW_WIDTH)
            density = gaussian.evaluate(x) * len(y)
            result[elem] = density
        return result

class RegionIdentifier(BaseProcessor):
    def __init__(self):
        self.cutoff_ratio_film = PARAMS.PLOT.ATOMDENSITY.CUTOFF_RATIO_FILM
        self.cutoff_ratio_mixed = PARAMS.PLOT.ATOMDENSITY.CUTOFF_RATIO_MIXED

    def check_layer_status(self, ratio):
        # Step 1: film exists?
        peak_val = np.max(ratio)
        if peak_val < self.cutoff_ratio_mixed:
            return peak_val, False, False
        elif self.cutoff_ratio_mixed < peak_val < self.cutoff_ratio_film:
            return peak_val, True, False
        elif self.cutoff_ratio_film < peak_val:
            return peak_val, True, True
        else:
            raise ValueError("Invalid cutoff values")

    def get_ratio(self, profile):
        eps = 1e-8  # avoid divide by zero
        CHFs = profile.get('C', 0) + profile.get('H', 0) + profile.get('F', 0)
        SiOs = profile.get('Si', 0) + profile.get('O', 0) + profile.get('N', 0) + eps
        total = CHFs + SiOs
        ratio = CHFs / total
        normalized_ratio = total / np.max(total)
        return ratio, normalized_ratio

class MixedRegionIdentifier(RegionIdentifier):
    def __init__(self, name, suffix):
        super().__init__()
        self.name = name
        self.filename_suffix = suffix

    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, profiles):
        x, y = [], []
        for key, profile in profiles.items():
            result = self.run_single(profile)
            value = [result[i] for i in ['z_min', 'z_max', 'h']]
            x.append(key)
            y.append(value)
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        labels = ['key', 'z_min(A)', 'z_max(A)', 'h(A)']
        return x, y, labels

    def run_single(self, profile):
        z_min, z_max, h = self.get_region_data(profile)
        result = { 'z_min': z_min, 'z_max': z_max, 'h': h }
        return result

    def get_region_data(self, profile):
        ratio, _ = self.get_ratio(profile)
        _, flag, _ = self.check_layer_status(ratio)
        if not flag:
            return None, None, 0.0

        z = profile['z']
        z_max = z[np.argmax(ratio)]
        mask_mixed = (z <= z_max) \
                     & (ratio >= self.cutoff_ratio_mixed) \
                     & (ratio < self.cutoff_ratio_film)
        if np.sum(mask_mixed) == 0:
            return None, None, 0.0

        z_mixed = z[mask_mixed]
        z_mixed_min = np.min(z_mixed)
        z_mixed_max = np.max(z_mixed)
        h_mixed = z_mixed_max - z_mixed_min  # mixed layer thickness
        return z_mixed_min, z_mixed_max, h_mixed

class FilmRegionIdentifier(RegionIdentifier):
    def __init__(self, name, suffix):
        super().__init__()
        self.name = name
        self.filename_suffix = suffix
        self.mri = MixedRegionIdentifier(name, suffix)

    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, profiles):
        x, y = [], []
        for key, profile in profiles.items():
            result = self.run_single(profile)
            value = [result[i] for i in ['z_min', 'z_max', 'h']]
            x.append(key)
            y.append(value)
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        labels = ['key', 'z_min(A)', 'z_max(A)', 'h(A)']
        return x, y, labels

    def run_single(self, profile):
        z_min, z_max, h = self.get_region_data(profile)
        result = { 'z_min': z_min, 'z_max': z_max, 'h': h, }
        return result

    def get_region_data(self, profile):
        ratio, normalized_ratio = self.get_ratio(profile)
        _, _, flag = self.check_layer_status(ratio)
        if not flag:
            return None, None, 0.0

        z = profile['z']
        _, z_mixed_max, _ = self.mri.get_region_data(profile)
        if z_mixed_max is None:
            z_mixed_max = 0.0
        mask_film = (z > z_mixed_max) \
                    & (ratio >= self.cutoff_ratio_film) \
                    & (normalized_ratio >= self.cutoff_ratio_film)
        if np.sum(mask_film) == 0:
            return None, None, 0.0

        z_film = z[mask_film]
        z_film_min = np.min(z_film)
        z_film_max = np.max(z_film)
        h_film = z_film_max - z_film_min

        return z_film_min, z_film_max, h_film

class NeighborBasedProcessor(BaseProcessor):
    def __init__(self, name, suffix, system=None):
        self.name = name
        self.filename_suffix = suffix
        self.system = system

        if system == 'SiO2':
            self.symbol_map = {'Si': 0, 'O': 1, 'C': 2, 'H': 3, 'F': 4}
        elif system == 'Si3N4':
            self.symbol_map = {'Si': 0, 'N': 1, 'C': 2, 'H': 3, 'F': 4}
        else:
            raise ValueError(f"Unknown system: {system}")

        self.cutoff_matrix = np.loadtxt(PARAMS.PLOT.ATOMDENSITY.path_cutoff_matrix)
        self.neighbor_extractor = NeighborInfoExtractor(
            cutoff_matrix=self.cutoff_matrix,
            symbol_to_index=self.symbol_map
        )

class SpxRatioProcessor(NeighborBasedProcessor):
    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images, height_dict):
        x, y = [], []
        for key, image in images.items():
            z_min, z_max, _ = height_dict.get(key, (None, None, None))
            spx_ratio = self.run_single(image, z_min, z_max)
            x.append(key)
            y.append([
                spx_ratio['sp3'],
                spx_ratio['sp2'],
                spx_ratio['sp'],
                spx_ratio['others'],
            ])
            print(f'{key}: sp3={spx_ratio["sp3"]}, sp2={spx_ratio["sp2"]}, '
                  f'sp={spx_ratio["sp"]}, others={spx_ratio["others"]}')
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        labels = ['key', 'sp3', 'sp2', 'sp', 'others']
        return x, y, labels

    def run_single(self, atoms, z_min, z_max):
        result = {'sp3': 0, 'sp2': 0, 'sp': 0, 'others': 0}
        if z_min is None or z_max is None:
            return result

        positions = atoms.get_positions()
        z_coords = positions[:, 2]
        z_mask = (z_coords >= z_min) & (z_coords <= z_max)
        if not np.any(z_mask):
            return result

        symbols = atoms.get_chemical_symbols()
        region_idx = np.where(z_mask)[0]
        carbon_idx = [i for i in region_idx if symbols[i] == 'C']
        if len(carbon_idx) == 0:
            return result

        nbrs_dict = self.neighbor_extractor.get_neighbors_of_type(
            atoms=atoms,
            center_idxs=carbon_idx,
            target_symbols=None
        )

        neighbor_counts = [len(nbrs_dict[i]) for i in carbon_idx]

        counts = np.array(neighbor_counts)
        result['sp3'] = int(np.count_nonzero(counts == 4))
        result['sp2'] = int(np.count_nonzero(counts == 3))
        result['sp']  = int(np.count_nonzero(counts == 2))
        result['others'] = int(np.count_nonzero(~np.isin(counts, [2, 3, 4])))

        return result

class CarbonNeighborProcessor(NeighborBasedProcessor):
    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images):
        carbons_per_snapshot = {}
        for key, atoms in images.items():
            carbons_per_snapshot[key] = self.run_single(atoms)
            print(f'Carbon classification for snapshot {key} done')

        all_bondtypes = set()
        for data in carbons_per_snapshot.values():
            all_bondtypes.update(data.keys())
        labels = sorted(all_bondtypes)

        result = {bt: [] for bt in labels}
        for key, data in carbons_per_snapshot.items():
            for bt in labels:
                result[bt].append(data.get(bt, 0))

        x = np.array(list(carbons_per_snapshot.keys()), dtype=int)
        y = np.array([result[bt] for bt in labels]).T  # shape=(n_snapshots, n_bondtypes)
        return x, y, ['x'] + labels

    def run_single(self, atoms):
        symbols = atoms.get_chemical_symbols()
        carbon_idxs = [i for i, s in enumerate(symbols) if s == 'C']
        if len(carbon_idxs) == 0:
            return {}

        nbrs_dict = self.neighbor_extractor.get_neighbors_of_type(
            atoms=atoms,
            center_idxs=carbon_idxs,
            target_symbols=None
        )

        classification = {}
        for ci in carbon_idxs:
            nbr_idxs = nbrs_dict.get(ci, [])
            nbr_syms = [symbols[j] for j in nbr_idxs]

            bondtype = self.get_bondtype(nbr_syms)
            if bondtype not in classification:
                classification[bondtype] = []
            classification[bondtype].append(ci)

        return {bt: len(idxs) for bt, idxs in classification.items()}

    def get_bondtype(self, symbols) -> str:
        if self.system == 'SiO2':
            case_dict = PARAMS.PLOT.ATOMDENSITY.case_dict_SiO2
        elif self.system == 'Si3N4':
            case_dict = PARAMS.PLOT.ATOMDENSITY.case_dict_Si3N4
        else:
            raise ValueError(f"Unknown system: {self.system}")

        if symbols is None:
            return 'etc'
        if isinstance(symbols, str):
            symbols = symbols.split()

        key = tuple(sorted(set(symbols)))
        out = case_dict.get(key)
        if out == 'CX':
            count_C = symbols.count('C')
            return f"C{count_C}"
        return out if out is not None else 'etc'

class HydrogenEffectProcessor(NeighborBasedProcessor):
    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images, height_dict):
        x, y = [], []
        for key, atoms in images.items():
            z_min, z_max, _ = height_dict.get(key, (None, None, None))
            x.append(key)
            y.append(self.run_single(atoms, z_min, z_max))
            print(f'Carbon classification for snapshot {key} done')
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=float)
        labels = ['key', 'n_Si+n_C']
        return x, y, ['x'] + labels

    def run_single(self, atoms, z_min, z_max):
        if z_min is None or z_max is None:
            return None

        positions = atoms.get_positions()
        z_coords = positions[:, 2]
        z_mask = (z_coords >= z_min) & (z_coords <= z_max)
        if not np.any(z_mask):
            return None

        symbols = atoms.get_chemical_symbols()
        region_idx = np.where(z_mask)[0]
        carbon_idxs = [i for i in region_idx if symbols[i] == 'C']
        if len(carbon_idxs) == 0:
            return None

        nbrs_dict = self.neighbor_extractor.get_neighbors_of_type(
            atoms=atoms,
            center_idxs=carbon_idxs,
            target_symbols=None
        )

        result = {}
        for ci in carbon_idxs:
            nbr_idxs = nbrs_dict.get(ci, [])
            nbr_syms = [symbols[j] for j in nbr_idxs]

            n_Si = nbr_syms.count('Si')
            n_C = nbr_syms.count('C')
            result[ci] = n_Si + n_C

        average_count = np.mean(list(result.values()))

        return average_count

class NeighborInfoExtractor:
    def __init__(self, cutoff_matrix, symbol_to_index):
        self.cutoff = cutoff_matrix
        self.sym2idx = symbol_to_index

    def get_all_neighbors(self, atoms):
        """
        Returns:
            adjacency: {i: [j1, j2, ...], ...}
        """
        atoms.wrap()

        positions = atoms.get_positions()                     # shape=(N,3)
        symbols = atoms.get_chemical_symbols()               # length=N
        atom_types = np.array([self.sym2idx[s] for s in symbols])  # shape=(N,)

        box_lengths = atoms.get_cell().lengths()             # array([Lx, Ly, Lz])
        try:
            tree = cKDTree(positions, boxsize=box_lengths)
        except:
            positions[np.abs(positions) < 1e-5] = 0.0
            tree = cKDTree(positions, boxsize=box_lengths)

        max_cutoff = np.max(self.cutoff)
        pairs_coo = tree.sparse_distance_matrix(tree, max_cutoff, output_type='coo_matrix')
        row = pairs_coo.row    # i
        col = pairs_coo.col    # j
        dist = pairs_coo.data  # distance

        N = len(atoms)
        adjacency = {i: [] for i in range(N)}

        for i, j, d in zip(row, col, dist):
            if i == j:
                continue
            ti = atom_types[i]
            tj = atom_types[j]
            if d <= self.cutoff[ti, tj]:
                adjacency[i].append(j)
        for i in adjacency:
            adjacency[i] = list(set(adjacency[i]))
        return adjacency

    def get_neighbors_of_type(self, atoms, center_idxs, target_symbols):
        """
        Returns:
            neighbors_of_interest: {center_idx: [neighbor_idx, ...], ...}
        """
        adjacency = self.get_all_neighbors(atoms)
        symbols = atoms.get_chemical_symbols()

        neighbors_of_interest = {}
        for ci in center_idxs:
            nbrs = adjacency.get(ci, [])
            if target_symbols is None:
                neighbors_of_interest[ci] = nbrs.copy()
            else:
                filtered = [j for j in nbrs if symbols[j] in target_symbols]
                neighbors_of_interest[ci] = filtered
        return neighbors_of_interest

