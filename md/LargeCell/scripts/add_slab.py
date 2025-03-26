import argparse
import numpy as np
from ase.io import read, write
# from functools import partial

class AtomStructure:
    def __init__(self, file_path, is_base=False):
        self.atom = self.read_file(file_path)
        self.is_base = is_base

    def read_file(self, file_path):
        read_options = {
            'format': 'lammps-data',
            'index': 0,
            'atom_style': 'atomic',
            'sort_by_id': False,
            'Z_of_type': {1: 14, 2: 8, 3: 6, 4: 1, 5: 9}
        }
        atom = read(file_path, **read_options)
        atom.wrap()
        return atom

class SlabModifier:
    def __init__(self, input_file, base_file, output_file, fix, thickness, criteria):
        self.input_structure = AtomStructure(input_file)
        self.base_structure = AtomStructure(base_file, is_base=True)
        self.output_file = output_file
        self.fix = fix
        self.thickness = thickness
        self.criteria = criteria

    def get_xyz_fixed_atoms(self):
        fix_atoms = np.array([atom.position for atom in self.input_structure.atom if atom.position[2] < self.fix])
        ind = np.lexsort((fix_atoms[:, 0], fix_atoms[:, 1], fix_atoms[:, 2]))
        return fix_atoms[ind]

    def check_match(self):
        '''
        Find the lowest atom of fixed layer in slab is consistent to bulk one
        (for those that does not match, skip)
        '''
        sorted_array = self.get_xyz_fixed_atoms()
        sel_atom_xy = sorted_array[:, :2]
        base_xy = self.base_structure.atom.positions[:, :2]

        matching_atoms = []
        for xy_coord in sel_atom_xy:
            is_atom = np.any(np.all((np.abs(base_xy - xy_coord) < 1e-3), axis=1))
            if is_atom:
                matching_atoms.append(xy_coord)

        if not matching_atoms:
            raise ValueError(f"Error occurred: no matching atoms found between input and base files")

        lowest_matching_atom = min(matching_atoms, key=lambda x: x[1])
        idx_lowest = np.nonzero(np.all((np.abs(base_xy - lowest_matching_atom) < 1e-3), axis=1))[0][0]

        return idx_lowest

    def shift_slab(self):
        '''
        Shift the slab to the top of the fixed layer
        '''
        shift_at = self.input_structure.atom.copy()
        z_coords = shift_at.positions[:, 2]
        shift_at = shift_at[z_coords > self.fix]
        shift_at.positions[:, 2] += self.thickness
        return shift_at

    def cut_bulk(self, base_idx):
        '''
        Cut the bulk atoms based on the height of the *base_idx* atom
        '''
        base_at = self.base_structure.atom
        max_z = base_at.cell[2, 2]
        min_z = base_at.positions[base_idx, 2] if base_idx is not None else 0

        z_pos_list = base_at.positions[:, 2]
        if min_z < self.thickness:
            check = lambda z: (z < min_z + self.fix) or (z > min_z - self.thickness + max_z)
        else:
            check = lambda z: (z < min_z + self.fix) and (z > min_z - self.thickness)

        mask = np.array([check(z) for z in z_pos_list])
        if base_idx is not None:
            mask[base_idx] = True

        filtered_atoms = base_at[mask]
        filtered_atoms.positions[:, 2] -= min_z - self.thickness
        filtered_atoms.wrap()
        return filtered_atoms

    def make_new_slab(self, end_idx):
        '''
        Make a new slab by shifting the slab and adding bulk atoms
        '''
        shift_at = self.shift_slab()
        add_at = self.cut_bulk(end_idx)
        mod_at = shift_at.copy()
        for atom in add_at:
            mod_at.append(atom)
        return mod_at

    def should_add_slab(self):
        '''
        Check if slab should be added
        based on penetration depth of etchant elements
        '''
        penetrate_elem_dict = ['C', 'H', 'F']
        min_z_list = []
        for element in penetrate_elem_dict:
            mask = [atom.index for atom in self.input_structure.atom if atom.symbol == element]
            pos_z = self.input_structure.atom.get_positions()[mask, 2]
            if len(pos_z) > 0:
                min_z_list.append(np.min(pos_z))

        if not min_z_list:
            print("No penetrated atoms, add_slab NOT NEEDED")
            return False

        min_z = min(min_z_list)
        if min_z < self.criteria:
            print(f"penet_depth: {min_z} < criteria: {self.criteria}, add_slab NEEDED")
            return True
        else:
            print(f"penet_depth: {min_z} < criteria: {self.criteria}, add_slab NOT NEEDED")
            return False

    def modify_slab(self):
        '''
        Add bulk atoms to the slab if needed
        '''
        if self.should_add_slab():
            end_idx = self.check_match()
            mod_at = self.make_new_slab(end_idx)
            write(
                self.output_file, mod_at, format='lammps-data', atom_style='atomic',
                velocities=True, specorder=['Si', 'O', 'C', 'H', 'F']
            )
            print(f"Modified slab written to {self.output_file}")
        else:
            print("No modification needed")

def main():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('-i', metavar='input_file', type=str, required=True, help='the path to the input file')
    parser.add_argument('-o', metavar='output_file', type=str, required=True, help='the path to the output file')
    parser.add_argument('-b', metavar='base_file', type=str, required=True, help='the path to the base file')
    parser.add_argument('-f', metavar='fix', type=float, default=2, help='the thickness of layer that fixed')
    parser.add_argument('-t', metavar='thick', type=float, default=20, help='the thickness of layer that added')
    parser.add_argument('-c', metavar='criteria', type=float, default=15, help='criteria for whether to add slab')
    args = parser.parse_args()

    modifier = SlabModifier(args.i, args.b, args.o, args.f, args.t, args.c)
    modifier.modify_slab()

if __name__ == '__main__':
    main()
