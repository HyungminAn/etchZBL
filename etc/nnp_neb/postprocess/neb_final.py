import tempfile
import sys
from ase.io import read
from ase.io import write


def ReadLastPartFromDump(path_dump, lines_to_read=840):
    with open(path_dump, 'r') as f:
        lines = f.readlines()[-lines_to_read:]

    tf = tempfile.NamedTemporaryFile()

    with open(tf.name, 'w') as f:
        for line in lines:
            f.write(line)

    atoms = read(tf.name, format='lammps-dump-text')

    return atoms


def main(*args, **kwargs):
    initial_dat = args[1]
    path_dumps = args[2:]

    with open(initial_dat, 'r') as f:
        n_atoms = int(f.readlines()[2].split()[0])
    lines_to_read = n_atoms + 11

    dump_final = []
    for path_dump in path_dumps:
        dump = ReadLastPartFromDump(path_dump, lines_to_read=lines_to_read)

        dump_final.append(dump)
        print(path_dump, 'Done')

    write('dump_final.XDATCAR', dump_final, format='vasp-xdatcar')


main(*sys.argv)
