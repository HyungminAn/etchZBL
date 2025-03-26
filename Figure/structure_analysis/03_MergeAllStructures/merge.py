import os
import sys
import pickle
import bisect
from dataclasses import dataclass
import functools
import yaml

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write

@dataclass
class RWoptions:
    '''
    ASE read/write options
    '''
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


def cache_to_pkl(pkl_path):
    '''
    Decorator to cache the result of a function to a .pkl file
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the .pkl file already exists
            if os.path.exists(pkl_path):
                print(f"Loading cached result from {pkl_path}")
                with open(pkl_path, "rb") as f:
                    return pickle.load(f)

            # Call the original function and save the result
            print(f"Running {func.__name__} and caching result to {pkl_path}")
            result = func(*args, **kwargs)
            with open(pkl_path, "wb") as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator


class MergedCellGenerator:
    def run(self, src, idx_list):
        str_dicts = self.get_structure_dicts(src)
        added_slabs = self.get_added_slab_structures(str_dicts)
        merged_cells = self.merge_cells(str_dicts, added_slabs)
        merged_cells_with_unitcell = self.write_poscars_with_merged_cells(str_dicts,
                                                                     merged_cells,
                                                                     idx_list,
                                                                     write_poscars=True)

    @cache_to_pkl("str_dicts.pkl")
    def get_structure_dicts(self, src):
        '''
        From src, read all .coo files and return a dictionary of the form:
        '''
        str_dict = {}
        str_after_mod_dict = {}
        str_rm_dict = {}
        str_add_dict = {}
        str_sub_dict = {}
        str_save_dict = {}

        def strip_rm(name):
            return int(name.replace('rm_byproduct_str_shoot_', '').replace('.coo', ''))
        def strip_add(name):
            return int(name.replace('add_str_shoot_', '').replace('.coo', ''))
        def strip_sub(name):
            return int(name.replace('sub_str_shoot_', '').replace('.coo', ''))
        def strip_save(name):
            return int(name.replace('save_str_shoot_', '').replace('.coo', ''))
        def strip_after_mod(name):
            return int(name.replace('str_shoot_', '').replace('_after_mod.coo', ''))
        def strip_str(name):
            return int(name.replace('str_shoot_', '').replace('.coo', ''))

        for root, dirs, files in os.walk(src):
            dirs.sort()
            files.sort()

            for file in files:
                if not file.endswith('.coo'):
                    continue

                if 'rm_' in file:
                    selected_dict = str_rm_dict
                    key = strip_rm(file)
                elif 'add_' in file:
                    selected_dict = str_add_dict
                    key = strip_add(file)
                elif 'save_' in file:
                    selected_dict = str_save_dict
                    key = strip_save(file)
                elif '_after_mod' in file:
                    selected_dict = str_after_mod_dict
                    key = strip_after_mod(file)
                elif 'sub_' in file:
                    selected_dict = str_sub_dict
                    key = strip_sub(file)
                else:
                    selected_dict = str_dict
                    key = strip_str(file)
                value = os.path.join(root, file)
                selected_dict[key] = value

        result = {
                'str_dict': str_dict,
                'str_after_mod_dict': str_after_mod_dict,
                'str_rm_dict': str_rm_dict,
                'str_add_dict': str_add_dict,
                'str_save_dict': str_save_dict,
                'str_sub_dict': str_sub_dict,
                }
        return result

    def get_max_z(self, path_poscar):
        poscar = read(path_poscar, **RWoptions.read_opts)
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

    @cache_to_pkl("added_slab_structures.pkl")
    def get_added_slab_structures(self, str_dicts):
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
            poscar_diff = read(str_after_mod_dict[key], **RWoptions.read_opts)
            poscar_diff = self.cut_cell_under_h(poscar_diff, z_diff)
            result[key] = poscar_diff
        return result

    @cache_to_pkl("merged_cells.pkl")
    def merge_cells(self, str_dicts, added_slabs):
        '''
        Merge the added slab structures into a single structure,
        which can be used to create a merged cell.
        '''
        str_save_dict = str_dicts['str_save_dict']
        keys = set(added_slabs.keys()) | set(str_save_dict.keys())
        keys = sorted(keys)
        poscar_adds = [(key, added_slabs[key]) for key in keys if
                       added_slabs.get(key) is not None]
        poscar_subs = [(key, read(str_save_dict[key], **RWoptions.read_opts))
                       for key in keys if str_save_dict.get(key) is not None]

        dst = 'poscars'
        if not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)

        result = {}

        key, poscar_merged = poscar_lists[-1][0], poscar_lists[-1][1].copy()
        result[key] = poscar_merged.copy()
        cell = poscar_merged.get_cell()
        shift_z = cell[2, 2]
        for (key, poscar) in poscar_lists[:-1][::-1]:
            poscar.translate([0, 0, shift_z])
            shift_z += poscar.get_cell()[2, 2]
            cell[2, 2] = shift_z
            poscar_merged.set_cell(cell)
            poscar_merged.extend(poscar)
            print(shift_z, len(poscar), len(poscar_merged))
            result[key] = poscar_merged.copy()
        return result

    @cache_to_pkl("write_poscars_with_merged_cells.pkl")
    def write_poscars_with_merged_cells(self,
                                        str_dicts,
                                        merged_cells,
                                        idx_list,
                                        write_poscars=False):
        '''
        Merge the added slab structures into a single structure,
        with the unit cell extended in the z-direction.
        '''
        str_after_mod_dict = str_dicts['str_after_mod_dict']
        str_save_dict = str_dicts['str_save_dict']
        str_sub_dict = str_dicts['str_sub_dict']
        keys = sorted(merged_cells.keys())

        if write_poscars:
            dst = 'merged_poscars'
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)

        def find_min_larger_value(sorted_list, input_value):
            index = bisect.bisect_right(sorted_list, input_value)
            if index < len(sorted_list):
                return sorted_list[index]
            else:
                return None

        result = {
                'fix_height': {},
                'atoms': {},
                }

        fix_h = 6.0
        for i in idx_list:
            poscar = read(str_after_mod_dict[i], **RWoptions.read_opts)
            _selected_idx = find_min_larger_value(keys, i)
            if _selected_idx is not None:
                slab = merged_cells[_selected_idx].copy()

                cell = slab.get_cell()
                slab_z = cell[2, 2]
                poscar.positions[:, 2] += slab_z
                unitcell_z = poscar.get_cell()[2, 2]
                cell[2, 2] += unitcell_z
                slab.set_cell(cell)
                slab.extend(poscar)
            else:
                slab = poscar.copy()
                slab_z = 0.0

            if str_save_dict.get(i) is not None:
                save = read(str_save_dict[i], **RWoptions.read_opts)
                pos = slab.get_positions()
                pos[:, 2] += save.get_cell()[2, 2]
                slab.set_positions(pos)
                slab.extend(save)
                slab_z += save.get_cell()[2, 2]

            if write_poscars:
                write(f'{dst}/{i}.lammps', slab, **RWoptions.write_opts)
                print(f'{dst}/{i}.lammps Written')

            result['fix_height'][i] = slab_z + fix_h
            result['atoms'][i] = slab

            print(f'{i} Done')
        return result


class HeightPlotter:
    def run(self, merged_cells_with_unitcell):
        plot_data = self.analyze_height(merged_cells_with_unitcell)
        self.plot(plot_data)

    @cache_to_pkl("analyze_height.pkl")
    def analyze_height(self, merged_cells_with_unitcell):
        '''
        Analyze the height of the elements in the merged cells.
        '''
        def get_lowest_z_of_elments(slab):
            '''
            Get the lowest z-coordinate of the elements in the slab.
            '''
            target_elements = ['C', 'H', 'F']
            lowest_z = {}
            pos = slab.get_positions()
            for element in target_elements:
                mask = np.array([i for i, atom in enumerate(slab) if atom.symbol == element])
                if len(mask) == 0:
                    continue
                lowest_z[element] = np.min(pos[mask, 2])
            return lowest_z

        result = {
                'elem_lowest_z': {},
                'fix_z': {},
                }
        atoms_dict = merged_cells_with_unitcell['atoms']
        fix_h_dict = merged_cells_with_unitcell['fix_height']
        for key, value in atoms_dict.items():
            elem_lowest_z = get_lowest_z_of_elments(value)
            result['elem_lowest_z'][key] = elem_lowest_z
            result['fix_z'][key] = fix_h_dict[key]
            print(f'{key} Done')
        return result

    def plot(self, data):
        fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(8, 6))

        keys = sorted(data['fix_z'].keys())
        fix_z = np.array([data['fix_z'][key] for key in keys])

        ax.plot(keys, fix_z, label='fix_z', linestyle='--', color='black')

        target_elements = [('C', '#909090'), ('F', '#ff55ff')]
        y_lowest = {}
        for (element, color) in target_elements:
            lowest_z = [
                data['elem_lowest_z'][key][element]
                if data['elem_lowest_z'][key].get(element) is not None
                else None
                for key in keys
                ]
            ax.plot(keys, lowest_z, label=element, color=color)
            y_lowest[element] = np.array([z if z is not None else np.inf for z in lowest_z])

        ax.set_xlabel('incidence')
        ax.set_ylabel('z ($\AA$)')
        ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

        y_lowest = np.minimum(y_lowest['C'], y_lowest['F'])
        ax_diff.plot(keys, y_lowest - fix_z, label='Diff', linestyle='--', color='black')
        ax_diff.set_xlabel('incidence')
        ax_diff.set_ylabel('Dist from fixed layer ($\AA$)')
        ax_diff.set_ylim(0, 10)
        ax_diff.grid(axis='y')
        ax_diff.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

        fig.tight_layout()
        fig.savefig('result.png')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} input.yaml")
        sys.exit(1)
    path_input_yaml = sys.argv[1]
    with open(path_input_yaml, 'r') as f:
        input_yaml = yaml.safe_load(f)

    src = input_yaml['src']
    n_start = input_yaml['n_start']
    n_end = input_yaml['n_end']
    n_step = input_yaml['n_step']
    idx_list = list(range(n_start, n_end + 1, n_step))

    m = MergedCellGenerator()
    m.run(src, idx_list)
    p = HeightPlotter()
    p.run(m.write_poscars_with_merged_cells)
