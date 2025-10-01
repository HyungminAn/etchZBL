import os
import sys
from ase.io import read, write


def main():
    path_outcar = sys.argv[1]
    interval = 20
    outcar = read(path_outcar, index=':')

    os.makedirs('poscars')

    count = 0
    for image in outcar[::interval]:
        write(f'poscars/POSCAR_{count}', image, format='vasp')
        print(f"POSCAR_{count} written")
        count += interval


main()
