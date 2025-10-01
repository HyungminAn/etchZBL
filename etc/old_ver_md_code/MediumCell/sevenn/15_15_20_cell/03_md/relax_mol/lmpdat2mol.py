from ase.io import read
from ase.data import atomic_masses as mass_dict


def main():
    molecule_list = ['CF', 'CF2', 'CF3', 'CHF', 'CH2F', 'CHF2']

    match_dict = {
            'C': (12.011, 3),
            'H': (1.008, 4),
            'F': (18.998, 5)
            }

    for mol in molecule_list:
        image = read(f'{mol}/relaxed.coo',
                     format='lammps-data', style='atomic')
        pos = image.get_positions()
        masses = [match_dict[i][0] for i in image.get_chemical_symbols()]
        atomic_numbers = [match_dict[i][1] for i in image.get_chemical_symbols()]
        n_atoms = len(image)

        with open(f'mol_{mol}', 'w') as f:
            w = f.write

            w(f"# {mol}\n\n")
            w(f"{n_atoms} atoms\n\n")

            w("Coords\n\n")
            for idx, (atom, xyz) in enumerate(zip(atomic_numbers, pos)):
                w(f"{idx+1} {xyz[0]} {xyz[1]} {xyz[2]}\n")
            w("\n")

            w("Types\n\n")
            for idx, atomic_number in enumerate(atomic_numbers):
                w(f"{idx+1} {atomic_number}\n")
            w("\n")

            w("Masses\n\n")
            for idx, atomic_number in enumerate(atomic_numbers):
                w(f"{idx+1} {mass_dict[atomic_number]}\n")

        print(f"{mol} Done")


main()
