import sys
import random
import subprocess
import numpy as np
from numpy.linalg import norm as norm
import datetime
# import pdb
"""
Usage: python -.py *path_output*

Modified by HM: 20230214
"""


def dist_pbc(v_a, v_b, cell):
    '''
    Distance wigh PBC
    '''
    imagecell = [-1, 0, 1]
    pbc = [
        [i, j, k]
        for i in imagecell
        for j in imagecell
        for k in imagecell
          ]
    v_b_sets = np.array([
        [v_b[i] + cell * pbc_subset[i] for i in range(3)]
        for pbc_subset in pbc
    ])

    return min([norm(v_a - v) for v in v_b_sets])


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
        newatom_type = chemical_symbols[idx_atom]
        newatom_pos = [cellpar * random.random() for i in range(3)]

        # skip for the very first atom
        idx_exatom = -1
        while idx_exatom < idx_atom:
            tot_attempt += 1

            # Exit Loop if it takes too long
            if tot_attempt > max_iter:
                sys.exit()

            newatom_pos = [cellpar * random.random() for i in range(3)]
            for idx_exatom in range(idx_atom + 1):
                exatom_type = chemical_symbols[idx_exatom]
                dist = dist_pbc(
                       newatom_pos,
                       existing_atom_pos[idx_exatom],
                       cellpar
                       )

                cutoff = cutoff_dict[(newatom_type, exatom_type)]
                cond = dist < cutoff
                if cond:
                    break

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
        w("{:21.13f}{:19.13f}{:19.13f}\n".format(cellpar, 0, 0))
        w("{:21.13f}{:19.13f}{:19.13f}\n".format(0, cellpar, 0))
        w("{:21.13f}{:19.13f}{:19.13f}\n".format(0, 0, cellpar))
        w(f'   {composition_data_wide}\n')
        w(f'   {n_atom_lists_data}\n')
        w('Selective dynamics\n')
        w('Direct\n')

        existing_atom_pos = [
            [existing_atom_pos[i][j] / cellpar for j in range(3)]
            for i in range(n_totatom)
        ]

        for i in range(n_totatom-1):
            x, y, z = existing_atom_pos[i]
            w(f"{x:19.15f}{y:19.15f}{z:19.15f}   T   T   T\n")

        # Fix for the last atom
        x, y, z = existing_atom_pos[n_totatom-1]
        w(f"{x:19.15f}{y:19.15f}{z:19.15f}   F   F   F\n")


def main(*args, **kwargs):
    # POTCAR location ; read atom mass using POTCAR
    potdir = '/data/vasp4us/pot/PBE54'

    # Inputs (about amorphous state)
    # density : 90% of calculated density (unit: g/cm3)
    density = 2.47692
    atomname = ['Si', 'O_s', 'C_s', 'H_s', 'F_s']
    n_atom_list = [21, 42, 14, 7, 7]
    n_element = len(atomname)
    n_totatom = sum(n_atom_list)

    chemical_symbols = [
        [i for k in range(j)]
        for i, j in zip(atomname, n_atom_list)
    ]
    chemical_symbols = [
        item for sublist in chemical_symbols
        for item in sublist
    ]
    # result : ['A', 'A', ..., 'A', 'B', ..., 'B', 'C', ...]

    # atomic cutoff radius (in the form of N x N list)
    cutoff_dict = {
        (element1, element2): 1.5
        for element1 in atomname
        for element2 in atomname
        }

    # Preallocate & grep MASS from POTCAR
    existing_atom_pos = [[0 for i in range(3)] for j in range(n_totatom)]
    atommass = [0.0 for i in range(n_element)]
    for i in range(n_element):
        atom_type = atomname[i]
        command_findmass = f"grep MASS {potdir}/{atom_type}/POTCAR"
        command_findmass += ' | awk \'{print $3}\' | cut -d\";\" -f1'
        atommass[i] = float(
                subprocess.getoutput(command_findmass)
                )

    total_mass = sum([a * b for a, b in zip(atommass, n_atom_list)])
    conversion_factor = 10.0 / 6.022
    cellpar = (total_mass / density * conversion_factor) ** (1.0 / 3.0)

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
    path_output = args[1]
    write_poscar(
            path_output=path_output,
            atomname=atomname,
            n_atom_list=n_atom_list,
            density=density,
            cellpar=cellpar,
            n_totatom=n_totatom,
            existing_atom_pos=existing_atom_pos,
            )


main(*sys.argv)
