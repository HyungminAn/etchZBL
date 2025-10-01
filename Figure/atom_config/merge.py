import os
import re
import sys
import pickle
import bisect
from dataclasses import dataclass
from functools import wraps
import yaml

import numpy as np

from ase.io import read, write

@dataclass
class RWoptions:
    '''
    ASE read/write options
    '''
    @dataclass
    class SiO2:
        read_opts = {
                'format': 'lammps-data',
                'Z_of_type': {
                    1: 14,
                    2: 8,
                    3: 6,
                    4: 1,
                    5: 9
                    },
                }
        write_opts = {
                'format': 'lammps-data',
                'specorder': ['Si', 'O', 'C', 'H', 'F'],
                }
    @dataclass
    class Si3N4:
        read_opts = {
                'format': 'lammps-data',
                'Z_of_type': {
                    1: 14,
                    2: 7,
                    3: 6,
                    4: 1,
                    5: 9
                    },
                }
        write_opts = {
                'format': 'lammps-data',
                'specorder': ['Si', 'N', 'C', 'H', 'F'],
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
                pkl_path = func_gen_name(self)
                if os.path.exists(pkl_path):
                    print(f"{pkl_path} already exists, loading data from it.")
                    with open(pkl_path, 'rb') as f:
                        # Load the data from the pickle file
                        return pickle.load(f)
                # Call the original function and save the result
                print(f"Running {func.__name__} and caching result to {pkl_path}")
                result = func(self, *args, **kwargs)
                with open(pkl_path, "wb") as f:
                    pickle.dump(result, f)
                return result
            return wrapper
        return decorator

# def cache_to_pkl(pkl_path):
#     '''
#     Decorator to cache the result of a function to a .pkl file
#     '''
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             # Check if the .pkl file already exists
#             if os.path.exists(pkl_path):
#                 print(f"Loading cached result from {pkl_path}")
#                 with open(pkl_path, "rb") as f:
#                     return pickle.load(f)

#             # Call the original function and save the result
#             print(f"Running {func.__name__} and caching result to {pkl_path}")
#             result = func(*args, **kwargs)
#             with open(pkl_path, "wb") as f:
#                 pickle.dump(result, f)
#             return result
#         return wrapper
#     return decorator

class StructureLoader:
    def __init__(self, name):
        self.name = name
        self.filename_suffix = 'str_dicts.pkl'

    @pklSaver.run(lambda self: f"{self.name}_{self.filename_suffix}")
    def run(self, src_list):
        """
        Scan directories in src_list for .coo files (excluding 'anneal'),
        categorize by filename patterns, and return dicts of file paths.
        """
        # Prepare dictionaries for each category
        categories = {
            'str_dict': {},
            'str_after_mod_dict': {},
            'str_rm_dict': {},
            'str_add_dict': {},
            'str_save_dict': {},
            'str_sub_dict': {},
        }

        # Regex patterns mapping to category keys and group for ID
        patterns = [
            (re.compile(r'rm_byproduct_str_shoot_(\d+)\.coo$'), 'str_rm_dict'),
            (re.compile(r'add_str_shoot_(\d+)\.coo$'), 'str_add_dict'),
            (re.compile(r'save_str_shoot_(\d+)\.coo$'), 'str_save_dict'),
            (re.compile(r'sub_str_shoot_(\d+)\.coo$'), 'str_sub_dict'),
            (re.compile(r'str_shoot_(\d+)_after_mod\.coo$'), 'str_after_mod_dict'),
        ]
        default_pattern = re.compile(r'str_shoot_(\d+)\.coo$')

        for src in src_list:
            for root, dirs, files in os.walk(src):
                dirs.sort()
                for file in sorted(files):
                    if not file.endswith('.coo') or 'anneal' in file:
                        continue

                    rel_path = os.path.join(root, file)
                    placed = False

                    # Try matching each pattern
                    for regex, cat in patterns:
                        match = regex.search(file)
                        if match:
                            key = int(match.group(1))
                            categories[cat].setdefault(key, rel_path)
                            placed = True
                            break

                    # Default category if no specific pattern matched
                    if not placed:
                        match = default_pattern.search(file)
                        if match:
                            key = int(match.group(1))
                            categories['str_dict'].setdefault(key, rel_path)

        return categories

class AddedStructureIdentifier:
    def __init__(self, RWoptions, name):
        self.RWoptions = RWoptions
        self.name = name
        self.filename_suffix = 'added_slab_structures.pkl'

    @pklSaver.run(lambda self: f"{self.name}_{self.filename_suffix}")
    def run(self, str_dicts):
        '''
        By comparing the structures in str_after_mod_dict and str_rm_dict,
        return the added slab structures (the difference between the two).
        '''
        str_add_dict = str_dicts['str_add_dict']
        keys = sorted(str_add_dict.keys())

        result = {}
        str_after_mod_dict = str_dicts['str_after_mod_dict']
        str_rm_dict = str_dicts['str_rm_dict']
        for key in keys:
            max_z_before = self.get_max_z(str_rm_dict[key])
            max_z_after = self.get_max_z(str_after_mod_dict[key])
            z_diff = max_z_after - max_z_before
            print(key, z_diff)
            poscar_diff = read(str_after_mod_dict[key], **self.RWoptions.read_opts)
            poscar_diff = self.cut_cell_under_h(poscar_diff, z_diff)
            result[key] = poscar_diff
        return result

    def get_max_z(self, path_poscar):
        poscar = read(path_poscar, **self.RWoptions.read_opts)
        poscar.wrap()
        return np.max(poscar.get_positions())

    def cut_cell_under_h(self, poscar, h):
        '''
        Cut the unit cell under the height h.
        '''
        poscar_cut = poscar.copy()
        while poscar_cut:
            poscar_cut.pop()
        for atom in poscar:
            if atom.position[2] <= h:
                poscar_cut.append(atom)
        cell = poscar_cut.get_cell()
        cell[2, 2] = h
        poscar_cut.set_cell(cell)
        return poscar_cut

class CellMerger:
    def __init__(self, RWoptions, name):
        self.RWoptions = RWoptions
        self.name = name
        self.filename_suffix = 'merged_cells.pkl'

    @pklSaver.run(lambda self: f"{self.name}_{self.filename_suffix}")
    def run(self, str_dicts, added_slabs):
        '''
        Merge the added slab structures into a single structure,
        which can be used to create a merged cell.
        '''
        str_save_dict = str_dicts['str_save_dict']
        keys = set(added_slabs.keys()) | set(str_save_dict.keys())
        keys = sorted(keys)
        poscar_adds = [
            (key, added_slabs[key]) for key in keys
            if added_slabs.get(key) is not None]
        # poscar_subs = [
        #     (key, read(str_save_dict[key], **self.RWoptions.read_opts))
        #     for key in keys if str_save_dict.get(key) is not None]

        key, poscar_merged = poscar_adds[-1][0], poscar_adds[-1][1].copy()
        result = {}
        result[key] = poscar_merged.copy()
        cell = poscar_merged.get_cell()
        shift_z = cell[2, 2]
        for (key, poscar) in poscar_adds[::-1][1:]:
            poscar.translate([0, 0, shift_z])
            shift_z += poscar.get_cell()[2, 2]
            cell[2, 2] = shift_z
            poscar_merged.set_cell(cell)
            poscar_merged.extend(poscar)
            print(shift_z, len(poscar), len(poscar_merged),
                  poscar.get_cell()[2, 2], poscar_merged.get_cell()[2, 2])
            result[key] = poscar_merged.copy()
        return result

class MergedPoscarWriter:
    def __init__(self, RWoptions, name):
        self.RWoptions = RWoptions
        self.name = name
        self.filename_suffix = 'write_poscars_with_merged_cells.pkl'

    @pklSaver.run(lambda self: f"{self.name}_{self.filename_suffix}")
    def run(self, str_dicts, merged_cells, idx_list,
            write_poscars=False, fix_h=6.0):
        '''
        Merge the added slab structures into a single structure,
        with the unit cell extended in the z-direction.
        '''
        str_after_mod_dict = str_dicts['str_after_mod_dict']
        str_save_dict = str_dicts['str_save_dict']
        str_sub_dict = str_dicts['str_sub_dict']
        keys = sorted(merged_cells.keys())

        dst = f'merged_poscars/{self.name}'
        if write_poscars and not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)

        if str_save_dict:
            start_of_slab_subtract = min(str_save_dict.keys())
        else:
            start_of_slab_subtract = float('inf')

        for i in idx_list:
            file_path = str_after_mod_dict.get(i)
            if file_path is None or not os.path.exists(file_path):
                print(f"File {file_path} does not exist, skipping.")
                continue
            poscar = read(file_path, **self.RWoptions.read_opts)
            _selected_idx = self.find_min_larger_value(keys, i)
            print(f'Processing {i}, selected idx: {_selected_idx}')
            if _selected_idx is not None:
                slab = merged_cells[_selected_idx].copy()
                self.combine_two_cells(poscar, slab)
            else:
                slab = poscar.copy()

            slab = self.patch_subtract_poscars(
                start_of_slab_subtract, slab,
                str_sub_dict, str_save_dict, i)

            if write_poscars:
                write(f'{dst}/{i}.lammps', slab, **self.RWoptions.write_opts)
                print(f'{dst}/{i}.lammps Written')

            print(f'{i} Done')

    def combine_two_cells(self, poscar_src, poscar_dst):
        '''
        Combine two cells by extending the z-dimension.
        '''
        cell = poscar_dst.get_cell()
        slab_z = cell[2, 2]
        poscar_src.positions[:, 2] += slab_z
        unitcell_z = poscar_src.get_cell()[2, 2]
        cell[2, 2] += unitcell_z
        poscar_dst.set_cell(cell)
        poscar_dst.extend(poscar_src)

    def find_min_larger_value(self, sorted_list, input_value):
        index = bisect.bisect_right(sorted_list, input_value)
        if index < len(sorted_list):
            return sorted_list[index]
        else:
            return None

    def patch_subtract_poscars(self,
                               start_of_slab_subtract,
                               slab,
                               str_sub_dict,
                               str_save_dict,
                               i):
        if i < start_of_slab_subtract:
            return slab

        sub_keys = sorted(str_sub_dict.keys(), reverse=True)
        for sub_key in sub_keys:
            if sub_key > i:
                continue
            save = read(str_save_dict[sub_key], **self.RWoptions.read_opts)

            cell = slab.get_cell()
            cell[2, 2] += save.get_cell()[2, 2]
            slab.set_cell(cell)

            pos = slab.get_positions()
            pos[:, 2] += save.get_cell()[2, 2]
            slab.set_positions(pos)

            slab.extend(save)
        return slab

class ExtxyzWriter:
    def __init__(self, RWoptions, name):
        self.RWoptions = RWoptions
        self.name = name

    def run(self, src='merged_poscars'):
        src = f'merged_poscars/{self.name}'
        files = [i for i in os.listdir(src) if i.endswith('.lammps')]
        files.sort(key=lambda x: int(x.split('.')[0]))
        files = [read(os.path.join(src, f), **self.RWoptions.read_opts) for f in files]
        dst = f'{self.name}_merged_poscars.extxyz'
        write(dst, files, format='extxyz')
        print(f'{dst} Written with {len(files)} structures')

class MergedCellGenerator:
    def __init__(self, system, name):
        if system == 'SiO2':
            self.RWoptions = RWoptions.SiO2
        elif system == 'Si3N4':
            self.RWoptions = RWoptions.Si3N4
        else:
            raise ValueError(f"Unsupported system: {system}")
        self.name = name

    def run(self, src, idx_list):
        sl = StructureLoader(self.name)
        str_dicts = sl.run(src)

        asi = AddedStructureIdentifier(self.RWoptions, self.name)
        added_slabs = asi.run(str_dicts)

        cm = CellMerger(self.RWoptions, self.name)
        merged_cells = cm.run(str_dicts, added_slabs)

        mpcw = MergedPoscarWriter(self.RWoptions, self.name)
        mpcw.run(str_dicts, merged_cells, idx_list, write_poscars=True)

        ew = ExtxyzWriter(self.RWoptions, self.name)
        ew.run()

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.yaml <SiO2|Si3N4>")
        sys.exit(1)

    path_input_yaml = sys.argv[1]
    system = sys.argv[2]
    with open(path_input_yaml, 'r') as f:
        input_yaml = yaml.safe_load(f)

    n_start = input_yaml['n_start']
    n_end = input_yaml['n_end']
    n_step = input_yaml['n_step']
    idx_list = list(range(n_start, n_end + 1, n_step))

    src_dict = input_yaml['src']
    for ion in src_dict.keys():
        for energy, path in src_dict[ion].items():
            src = path
            mcg = MergedCellGenerator(system, f'{ion}_{energy}')
            mcg.run(src, idx_list)


if __name__ == "__main__":
    main()
