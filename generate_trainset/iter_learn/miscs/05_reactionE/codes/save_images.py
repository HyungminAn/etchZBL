import ase
import ase.neighborlist
from ase.io import read, write
import ase.build
import pickle


def main():
    with open("./to_rlx.dat",'r') as f:
        save_list=[
            sorted([int(image_idx) for image_idx in line.split()[1:]])
            for line in f.readlines()[1:]
        ]

    coos = []
    # nions = 50
    nions = 20

    read_options = {
        "format": "lammps-data",
        "index": 0,
        "atom_style": "atomic"
    }

    for i in range(1, nions+1):
        for j in range(len(save_list[i-1])):
            src = f"./incidences/{i}/{i}_{save_list[i-1][j]}/rlx.coo"
            coos.append(read(src, **read_options))

    with open(f"nnp_rlx.pickle",'wb') as f:
        pickle.dump(coos, f)

    # matcher={1:14,2:7,3:1,4:9}

    for i in range(len(coos)):
        # coos[i].set_atomic_numbers([matcher[j] for j in coos[i].get_atomic_numbers()])
        ase.build.sort(coos[i])

    write("nnp_rlx.extxyz", coos, format='extxyz')


if __name__ == '__main__':
    main()
