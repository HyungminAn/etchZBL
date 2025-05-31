import os
from functools import wraps
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

from ase.data import atomic_numbers, atomic_masses

from imageloader import ImageLoader
from params import PARAMS

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
                    with open(path_save, 'r') as f:
                        header = f.readline().strip()
                        labels = header.split()[1:]
                    return x, y, labels
                x, y, labels = func(self, *args, **kwargs)
                header = f'{" ".join(labels)}'
                np.savetxt(path_save, np.c_[x, y], comments='# ', header=header)
                return x, y, labels
            return wrapper
        return decorator

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
    def __init__(self, name):
        self.name = name
        self.filename_suffix = 'shifted_height.txt'

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
        imgldr = ImageLoader()
        add_dict = imgldr.get_file_list(src_list, pattern='add_str_shoot_')
        sub_dict = imgldr.get_file_list(src_list, pattern='sub_str_shoot_')

        for key in add_dict.keys():
            y[x > key] -= PARAMS.PLOT.HEIGHT.SHIFT
        for key in sub_dict.keys():
            y[x > key] += PARAMS.PLOT.HEIGHT.SHIFT

        labels = 'key height(A)'
        return x, y, labels

class CarbonChangeProcessor(BaseProcessor):
    def __init__(self, name):
        self.name = name
        self.filename_suffix = 'carbonfilm.txt'

    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images):
        x, y = [], []
        for key, image in images.items():
            t_carbon = self.calculate(key, image)
            x.append(key)
            y.append(t_carbon)
        x, y = np.array(x), np.array(y)
        labels = 'key thickness(A)'
        return x, y, labels

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

class FilmAnalyzer(BaseProcessor):
    def __init__(self, name):
        self.name = name
        self.filename_suffix = 'thickness.txt'

    def run(self, images):
        path_save = f'{self.name}_{self.filename_suffix}'
        if os.path.exists(path_save):
            print(f"{path_save} already exists, loading data from it.")
            data = np.loadtxt(path_save, skiprows=1)
            x = data[:, 0].astype(float)
            y = data[:, 1:]
            labels = 'key h_mixed(A) h_film(A) ...'
            return x, y, labels

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
        labels = [
                'key',
                'h_mixed(A)', 'h_film(A)',
                'density_mixed_layer(g/cm^3)', 'density_film_layer(g/cm^3)',
                'fc_ratio_mixed_layer', 'fc_ratio_film_layer',
                'sp3_ratio_mixed_layer', 'sp2_ratio_mixed_layer',
                'sp_ratio_mixed_layer', 'others_ratio_mixed_layer',
                'sp3_ratio_film_layer', 'sp2_ratio_film_layer',
                'sp_ratio_film_layer', 'others_ratio_film_layer',
                ]
        SAVE_OPTS = {
            'header': f'{" ".join(labels)}',
            'comments': f'# ',
        }
        np.savetxt(path_save, mat, **SAVE_OPTS)
        return x, y, labels

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
        if len(mask_C) == 0 or len(mask_F) == 0:
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

class CarbonNeighborProcessor(BaseProcessor):
    """
    Classify carbon atoms based on neighbor atom types using a cutoff matrix.

    Attributes:
        atoms: ASE Atoms object
        cutoff_matrix: 2D numpy array of pairwise cutoffs (element_type x element_type)
        element_order: list of element symbols defining indices in cutoff_matrix
    """
    def __init__(self, name, element_order=None):
        self.name = name
        self.cutoff_matrix = np.loadtxt('cutoff_matrix.npy')
        # Default element order if not provided
        if element_order is None:
            self.element_order = ["Si", "O", "C", "H", "F"]
        else:
            self.element_order = element_order
        self.symbol_to_index = {sym: i for i, sym in enumerate(self.element_order)}
        self.filename_suffix = 'carbon_neighbors.txt'

    @pklSaver.run(lambda self: f'{self.name}_{self.filename_suffix}')
    def run(self, images):
        carbons = {}
        for key, image in images.items():
            carbons[key] = self.run_single(image)
            # print(f'{key}: {carbons}')

        result = {}
        for key, data in carbons.items():
            for bondtype in data.keys():
                if result.get(bondtype) is None:
                    result[bondtype] = []

        labels = [i for i in result.keys()]
        for key, data in carbons.items():
            for bondtype in labels:
                count = data.get(bondtype, 0)
                result[bondtype].append(count)
        x = np.array([i for i in carbons.keys()])
        y = np.array([i for i in result.values()]).T
        labels = ['x'] + labels
        return x, y, labels

    def run_single(self, atoms):
        """
        Classify each carbon atom based on its neighbors.
        Returns:
            dict: mapping carbon atom index -> classification string
        """
        adjacency = self.build_adjacency(atoms)
        symbols = atoms.get_chemical_symbols()
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

    def build_adjacency(self, atoms):
        """
        Build adjacency list based on cutoff distances with periodic boundary conditions.
        Returns:
            adjacency: list of sets, adjacency[i] is the set of neighbor indices of atom i
        """
        # Ensure wrapped within cell
        atoms.wrap()
        positions = atoms.get_positions()
        # Clean tiny negative values
        positions[positions < -1e-10] = 0.0
        # PBC box lengths
        cell_lengths = atoms.get_cell().lengths()
        # Build k-d tree
        tree = cKDTree(positions, boxsize=cell_lengths)
        max_cutoff = np.max(self.cutoff_matrix)
        # Sparse distance in COO
        pairs = tree.sparse_distance_matrix(tree, max_cutoff, output_type='coo_matrix')
        num_atoms = len(atoms)
        adjacency = [set() for _ in range(num_atoms)]
        row, col, dist = pairs.row, pairs.col, pairs.data
        atom_types = np.array([self.symbol_to_index[s] for s in atoms.get_chemical_symbols()])
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
