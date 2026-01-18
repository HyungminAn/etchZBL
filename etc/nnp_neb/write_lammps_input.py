import os
import sys
"""
Usage: python -.py *path_poscar*
"""


def get_atom_order(path_poscar):
    with open(path_poscar, 'r') as f:
        atom_order = f.readlines()[5]

    atom_list = ['Si', 'N', 'H', 'F']

    atoms = atom_order.strip('\n').split()
    if len(atoms) < len(atom_list):
        atoms += [i for i in atom_list if i not in atoms]

    atom_order = ' '.join(atoms)

    return atom_order


def write_lmp_input_relax(path_poscar, path_potential):
    atom_order = get_atom_order(path_poscar)

    atom_mass_dict = {
        'Si': 28.0855,
        'N': 14.0067,
        'H': 1.0078,
        'F': 14.007,
    }

    path_code = os.path.dirname(__file__)
    with open(f'{path_code}/relax_header.in', 'r') as f:
        lines_header = f.readlines()
    with open(f'{path_code}/relax_footer.in', 'r') as f:
        lines_footer = f.readlines()

    with open('relax.in', 'w') as f:
        w = f.write

        for line in lines_header:
            w(line)

        w('# Set mass\n')
        for idx, atom_type in enumerate(atom_order.split()):
            atom_mass = atom_mass_dict[atom_type]
            w(f'mass {idx+1} {atom_mass}\n')

        for line in lines_footer:
            if 'my_atom_order' in line:
                w(line.replace('my_atom_order', f'{atom_order}'))
                continue
            elif 'my_potential' in line:
                w(line.replace('my_potential', f'{path_potential}'))
                continue
            w(line)


def write_lmp_input_neb(path_poscar, path_potential):
    atom_order = get_atom_order(path_poscar)

    atom_mass_dict = {
        'Si': 28.0855,
        'N': 14.0067,
        'H': 1.0078,
        'F': 14.007,
    }

    path_code = os.path.dirname(__file__)
    with open(f'{path_code}/neb_header.in', 'r') as f:
        lines_header = f.readlines()
    with open(f'{path_code}/neb_footer.in', 'r') as f:
        lines_footer = f.readlines()

    with open('neb.in', 'w') as f:
        w = f.write

        for line in lines_header:
            w(line)

        for idx, atom_type in enumerate(atom_order.split()):
            atom_mass = atom_mass_dict[atom_type]
            w(f'mass {idx+1} {atom_mass}\n')

        for line in lines_footer:
            if 'my_atom_order' in line:
                w(line.replace('my_atom_order', f'{atom_order}'))
                continue
            elif 'my_potential' in line:
                w(line.replace('my_potential', f'{path_potential}'))
                continue
            w(line)


def main():
    path_poscar, path_potential = sys.argv[1], sys.argv[2]
    write_lmp_input_relax(path_poscar, path_potential)
    write_lmp_input_neb(path_poscar, path_potential)


main()
