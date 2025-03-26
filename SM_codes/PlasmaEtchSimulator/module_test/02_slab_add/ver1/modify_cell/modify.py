from ase.io import read, write


def main():
    atoms = read('str_shoot_0.coo', format='lammps-data')
    atoms_new = atoms.copy()
    while atoms_new:
        atoms_new.pop()
    for atom in atoms:
        if atom.position[2] < 25.0:
            atoms_new.append(atom)
    write('str_shoot_0_new.coo', atoms_new, format='lammps-data')


if __name__ == '__main__':
    main()
