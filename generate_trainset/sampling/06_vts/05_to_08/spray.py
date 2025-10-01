import os
from itertools import product
import subprocess
import datetime
import cProfile

import numpy as np
from numpy.linalg import norm as norm
"""
Usage: python -.py *path_output*

Modified by HM: 20230214
- Add fixed volume by HM: 20231102
"""


class MaxIterationExcessError(Exception):
    '''
    Exit Loop as it takes too long
    '''
    pass


def dist_pbc(v_a, v_b, cell, _pbc_cache={}):
    '''
    Distance wigh PBC
    '''
    pbc = _pbc_cache.get('pbc')
    if pbc is None:
        imagecell = [-1.0, 0.0, 1.0]
        pbc = np.array([i for i in product(imagecell, repeat=3)])
        for i in range(3):
            pbc[:, i] *= cell[i]
        _pbc_cache['pbc'] = pbc

    v_b_sets = pbc + v_b  # broadcast
    norms = norm(v_a - v_b_sets, axis=1)
    return min(norms)


def get_random_pos(cellpar):
    x, y, z = cellpar
    rnd_x, rnd_y, rnd_z = np.random.random(3)
    new_pos = np.array([x*rnd_x, y*rnd_y, z*rnd_z])
    return new_pos


def are_atoms_too_close(
        newatom_pos, newatom_type, idx_exatom, chemical_symbols,
        existing_atom_pos, cutoff_dict, cellpar):

    exatom_type = chemical_symbols[idx_exatom]
    dist = dist_pbc(newatom_pos, existing_atom_pos[idx_exatom], cellpar)

    cutoff = cutoff_dict[(newatom_type, exatom_type)]
    cond = dist < cutoff
    return cond


def find_safe_pos(
        idx_atom, cellpar, tot_attempt, max_iter, chemical_symbols,
        existing_atom_pos, cutoff_dict):
    '''
    For the given atom, find the position that satisfies cutoff condition.
    '''
    newatom_type = chemical_symbols[idx_atom]
    # skip for the very first atom
    newatom_pos = get_random_pos(cellpar)
    idx_exatom = -1
    while idx_exatom < idx_atom:
        tot_attempt += 1

        if tot_attempt > max_iter:
            raise MaxIterationExcessError

        newatom_pos = get_random_pos(cellpar)
        for idx_exatom in range(idx_atom + 1):
            cond = are_atoms_too_close(
                newatom_pos, newatom_type, idx_exatom, chemical_symbols,
                existing_atom_pos, cutoff_dict, cellpar)
            if cond:
                break

        # If all tests are passed, idx_exatom == idx_exatom

    return newatom_pos, tot_attempt


def random_spray(
        *args,
        n_totatom=None,
        n_element=None,
        chemical_symbols=None,
        cellpar=None,
        existing_atom_pos=None,
        cutoff_dict=None,
        max_iter=100000, **kwargs,
        ):
    '''
    Random spray

    Iteratively putting atoms
    '''
    tot_attempt = 0

    for idx_atom in range(n_totatom):
        newatom_pos, tot_attempt = find_safe_pos(
            idx_atom, cellpar, tot_attempt, max_iter, chemical_symbols,
            existing_atom_pos, cutoff_dict)

        print(f"Atom idx {idx_atom} : {newatom_pos}")
        existing_atom_pos[idx_atom] = newatom_pos
    #    print idx_atom , 'th atom'

    print(f'Total attempts : {tot_attempt}')
    print(f'Cell parameter : {cellpar}')

    return existing_atom_pos


def write_poscar(
        *args, path_output=None, atomname=None, n_atom_list=None,
        density=None, cellpar=None, n_totatom=None, existing_atom_pos=None,
        **kwargs):
    '''
    Writing POSCAR (Converting Fractional to Direct coordinate
    '''
    with open(path_output, 'w') as POSCAR:
        w = POSCAR.write

        time_data = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        composition_data = ''.join(atomname)
        composition_data_wide = '   '.join(atomname)
        n_atom_lists_data = '    '.join([str(x) for x in n_atom_list])

        w(f"{composition_data}   density: {density}   {time_data}\n")
        w("   1.0\n")
        w("{:21.13f}{:19.13f}{:19.13f}\n".format(cellpar[0], 0, 0))
        w("{:21.13f}{:19.13f}{:19.13f}\n".format(0, cellpar[1], 0))
        w("{:21.13f}{:19.13f}{:19.13f}\n".format(0, 0, cellpar[2]))
        w(f'   {composition_data_wide}\n')
        w(f'   {n_atom_lists_data}\n')
        w('Selective dynamics\n')
        w('Direct\n')

        existing_atom_pos = [
            [existing_atom_pos[i][j] / cellpar[j] for j in range(3)]
            for i in range(n_totatom)
        ]

        for i in range(n_totatom-1):
            x, y, z = existing_atom_pos[i]
            w(f"{x:19.15f}{y:19.15f}{z:19.15f}   T   T   T\n")

        # Fix for the last atom
        x, y, z = existing_atom_pos[n_totatom-1]
        w(f"{x:19.15f}{y:19.15f}{z:19.15f}   F   F   F\n")


def get_chemical_symbol_list(atomname, n_atom_list):
    '''
    result : ['A', 'A', ..., 'A', 'B', ..., 'B', 'C', ...]
    '''
    chemical_symbols = [
        [atom_type for _ in range(n_atoms)]
        for atom_type, n_atoms in zip(atomname, n_atom_list)
    ]
    chemical_symbols = [
        item for sublist in chemical_symbols
        for item in sublist
    ]
    return chemical_symbols


def get_cutoff_dict(atomname):
    # atomic cutoff radius (in the form of N x N list)
    cutoff_dict = {
        (element1, element2): 1.6
        for element1 in atomname
        for element2 in atomname
        }

    cutoff_dict[('H', 'H')] = 1.3

    return cutoff_dict


def get_atom_mass(n_element, atomname, potdir):
    atommass = [0.0 for i in range(n_element)]
    for i in range(n_element):
        atom_type = atomname[i]
        command_findmass = f"grep MASS {potdir}/{atom_type}/POTCAR"
        command_findmass += ' | awk \'{print $3}\' | cut -d\";\" -f1'
        atommass[i] = float(
                subprocess.getoutput(command_findmass)
                )
    return atommass


def calculate_cellpar(n_element, n_atom_list, atomname, density, potdir):
    atommass = get_atom_mass(n_element, atomname, potdir)
    total_mass = sum([a * b for a, b in zip(atommass, n_atom_list)])
    amu_to_gram = 1.66E-24
    angs3_to_cm3 = 1E-24
    # density = (total_mass * amu_to_gram) / (cell_volume * angs3_to_cm3)
    cell_volume = (total_mass * amu_to_gram) / (density * angs3_to_cm3)
    cell_par = cell_volume ** (1/3)
    print(f"Cell parameter : {cell_par}")
    return [cell_par] * 3


def spray(density, atomname, n_atom_list, path_output):
    # POTCAR location ; read atom mass using POTCAR
    potdir = '/data/vasp4us/pot/PBE54'

    n_element = len(atomname)
    n_totatom = sum(n_atom_list)
    chemical_symbols = get_chemical_symbol_list(atomname, n_atom_list)
    cutoff_dict = get_cutoff_dict(atomname)

    # Preallocate & grep MASS from POTCAR
    existing_atom_pos = np.zeros((n_totatom, 3))
    cellpar = calculate_cellpar(
        n_element, n_atom_list, atomname, density, potdir)

    # Random Spray
    random_spray(
        n_totatom=n_totatom,
        n_element=n_element,
        chemical_symbols=chemical_symbols,
        cellpar=cellpar,
        existing_atom_pos=existing_atom_pos,
        cutoff_dict=cutoff_dict,
        )

    # Write POSCAR
    write_poscar(
        path_output=path_output,
        atomname=atomname,
        n_atom_list=n_atom_list,
        density=density,
        cellpar=cellpar,
        n_totatom=n_totatom,
        existing_atom_pos=existing_atom_pos,
        )


def main():
    item_list = [
        [2.4667, ['Si', 'O', 'C', 'H', 'F'], [33, 33, 11, 11, 11]],
        [2.4667*0.5, ['Si', 'O', 'C', 'H', 'F'], [33, 33, 11, 11, 11]],
        [2.25, ['Si', 'O', 'C', 'H', 'F'], [6, 6, 6, 72, 6]],
        [2.25*0.5, ['Si', 'O', 'C', 'H', 'F'], [6, 6, 6, 72, 6]],
        [2.25, ['Si', 'O', 'C', 'H', 'F'], [8, 64, 8, 8, 8]],
        [2.25*0.5, ['Si', 'O', 'C', 'H', 'F'], [8, 64, 8, 8, 8]],
        [2.36, ['Si', 'O', 'F'], [20, 20, 60]],
        [2.36*0.5, ['Si', 'O', 'F'], [20, 20, 60]],
    ]
    path_output_list = [
        '01_density_normal/01_33111',
        '02_density_low/01_33111',
        '01_density_normal/02_050505605',
        '02_density_low/02_050505605',
        '01_density_normal/03_054050505',
        '02_density_low/03_054050505',
        '01_density_normal/04_11003',
        '02_density_low/04_11003',
    ]
    for (density, atomname, n_atom_list), path_output_dir in zip(
            item_list, path_output_list):
        os.makedirs(path_output_dir, exist_ok=True)
        path_output = f"{path_output_dir}/POSCAR"
        spray(density, atomname, n_atom_list, path_output)


cProfile.run("main()", "result.prof")
