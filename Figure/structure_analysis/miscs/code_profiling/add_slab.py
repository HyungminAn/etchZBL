from functools import partial
import argparse
import ase.io
import shutil
import numpy as np
import cProfile
"""
Original code by Sangmin Oh
Modified by Hyungmin An (2023. 11. 07)
"""


def read_args():
    parser = argparse.ArgumentParser(description="Process some files.")
    # Add the arguments
    parser.add_argument(
        '-i', metavar='input_file', type=str, required=True,
        help='the path to the input file'
        )
    parser.add_argument(
        '-o', metavar='output_file', type=str, required=True,
        help='the path to the output file'
        )
    parser.add_argument(
        '-b', metavar='base_file', type=str, required=True,
        help='the path to the base file'
        )
    parser.add_argument(
        '-f', metavar='fix', type=float,
        help='the thickness of layer that fixed', default=2
        )
    parser.add_argument(
        '-t', metavar='thick', type=float,
        help='the thickness of layer that added', default=20
        )
    parser.add_argument(
        '-c', metavar='criteria', type=float,
        help='criteria for whether to add slab', default=15
        )
    args = parser.parse_args()
    return args


def fetch_file(infile, basefile):
    '''
    Load as ase.Atom
    '''
    in_atom = ase.io.read(
        infile, format='lammps-data', index=0,
        style="atomic", sort_by_id=False, )
    base_atom = ase.io.read(
        basefile, format='lammps-data', index=0,
        style="atomic", sort_by_id=False, )

    return in_atom, base_atom


def get_xyz_fixed_atoms_in_slab(in_at, fix=3):
    fix_atoms = np.array(
        [atom.position for atom in in_at if atom.position[2] < fix]
        )
    ind = np.lexsort((fix_atoms[:, 0], fix_atoms[:, 1], fix_atoms[:, 2]))

    sorted_array = fix_atoms[ind]
    return sorted_array


def check_atom_match(base_xy, sel_atom_xy, infile, basefile, lowest_match):
    '''
    Find corresponding atom in bulk structure, for each fixed atom in slab
    '''
    atom_match = True

    for xy_coord in sel_atom_xy:
        is_atom = np.any(np.all((np.abs(base_xy - xy_coord) < 1e-3), axis=1))
        atom_match = is_atom and atom_match

    assert atom_match is True, f"\
        Error occured : the atoms of {infile} is not in {basefile},\
        check the files"


def check_match(in_at, base_at, infile, basefile, lowest_match=5, fix=3):
    '''
    Find the lowest atom of fixed layer in slab is consistent to bulk one

    sel_atom: atoms with lowermost z-coordination in slab
    '''
    sorted_array = get_xyz_fixed_atoms_in_slab(in_at, fix=fix)
    sel_atom_xy = sorted_array[:lowest_match, :2]
    base_xy = base_at.positions[:, :2]

    check_atom_match(base_xy, sel_atom_xy, infile, basefile, lowest_match)

    idx_lowest = np.all((np.abs(base_xy - sel_atom_xy[0, :]) < 1e-3), axis=1)
    idx_lowest = np.nonzero(idx_lowest)[0][0]

    return idx_lowest


def shift_slab(in_at, shift=10, fix=0):
    '''
    Shift unfixed atoms by *shift*
    '''
    shift_at = in_at.copy()
    if fix != 0:
        z_coords = shift_at.positions[:, 2]
        shift_at = shift_at[z_coords > fix]

    shift_at.positions[:, 2] += shift

    return shift_at


def get_min_z(in_at, base_idx):
    if base_idx is None:
        min_z = 0
    else:
        min_z = in_at.positions[base_idx, 2]
    return min_z


def filter1(z_pos, min_z, max_z, fix, cut):
    return (z_pos < min_z+fix) or (z_pos > min_z-cut + max_z)


def filter2(z_pos, min_z, fix, cut):
    return (z_pos < min_z+fix) and (z_pos > min_z-cut)


def get_mask(in_at, min_z, max_z, fix, cut, base_idx):
    z_pos_list = in_at.positions[:, 2]

    cond = min_z < cut
    if cond:
        check = partial(filter1, min_z=min_z, max_z=max_z, fix=fix, cut=cut)
    else:
        check = partial(filter2, min_z=min_z, fix=fix, cut=cut)

    mask = np.array([check(zpos) for zpos in z_pos_list])
    if base_idx is not None:
        mask[base_idx] = True

    return mask


def cut_bulk(base_at, cut=10, base_idx=None, shift=True, fix=0):
    '''
    cut: how much new slab will be added (in angstrom)
    base_idx: index of the atom in lowermost atom in slab,
        which corresponding in base structure
    '''
    max_z = base_at.cell[2, 2]
    min_z = get_min_z(base_at, base_idx)

    mask = get_mask(base_at, min_z, max_z, fix, cut, base_idx)
    filtered_atoms = base_at[mask]

    if shift:
        filtered_atoms.positions[:, 2] -= min_z-cut
        filtered_atoms.wrap()

    return filtered_atoms


def add_atoms(in_at, add_at):
    out_at = in_at.copy()
    for atom in add_at:
        out_at.append(atom)

    return out_at


def make_new_slab(in_at, base_at, put_thick, end_idx, fix):
    '''
    shift and cut to create slab
    '''
    shift_at = shift_slab(in_at, shift=put_thick, fix=fix)
    add_at = cut_bulk(base_at, cut=put_thick, base_idx=end_idx, fix=fix)
    mod_at = add_atoms(shift_at, add_at)
    return mod_at


def should_slab_be_added(in_at, cut_z):
    '''
    Here, atom match:
    '''
    penetrate_elem_dict = {
        'Li': 'C',
        'Be': 'H',
        'B': 'F',
    }
    min_z_list = []
    # Get penetration depth for each etchant element
    for element in penetrate_elem_dict.keys():
        mask = [atom.index for atom in in_at if atom.symbol == element]
        pos_z = in_at.get_positions()[mask, 2]
        # Skip for no element
        if len(pos_z) == 0:
            continue
        min_z = np.min(pos_z)
        min_z_list.append(min_z)

    if min_z_list:
        min_z = min(min_z_list)
    else:
        print(f"No penetrated atoms, add_slab NOT NEEDED")
        return False

    if min_z < cut_z:
        print(f"penet_depth: {min_z} < criteria: {cut_z}, add_slab NEEDED")
        return True
    else:
        print(f"penet_depth: {min_z} < criteria: {cut_z}, add_slab NOT NEEDED")
        return False


def main():
    args = read_args()
    put_thick = args.t
    fix = args.f
    infile = args.i
    basefile = args.b
    outfile = args.o
    criteria = args.c
    print(f"Input str : {infile} Base str : {basefile} \
            Out str : {outfile} with fix {fix} and thickness {put_thick}\
            with criteria : {criteria}")

    in_at, base_at = fetch_file(infile, basefile)

    to_add_new_slab = should_slab_be_added(in_at, criteria)

    if to_add_new_slab:
        end_idx = check_match(in_at, base_at, infile, basefile, fix=fix)
        mod_at = make_new_slab(in_at, base_at, put_thick, end_idx, fix)
        ase.io.write(
            outfile, mod_at, format='lammps-data', atom_style='atomic',
            velocities=True, specorder=["H", "He", "Li", "Be", "B"]
            )


if __name__ == '__main__':
    cProfile.run("main()", "result.prof")
