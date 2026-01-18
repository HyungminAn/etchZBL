import sys
from ase.io import read, write
"""
Usage: python -.py *XDATCAR* *idx1* *idx2*
"""


def main():
    path_dump, idx1, idx2 = sys.argv[1:4]
    idx1 = int(idx1)
    idx2 = int(idx2)
    dump = read(sys.argv[1], index=':')

    image1 = dump[idx1]
    write('POSCAR_initial', image1, format='vasp')
    image2 = dump[idx2]
    write('POSCAR_final', image2, format='vasp')


main()
