import time
import ase,os,sys
import numpy as np
import ase.neighborlist
import ase.io
import pickle
import graph_tool.all as gt

from graph_similarity_save_pickle import atom_image
"""
Original code by Changho Hong
"""


def save_rlx_image(Image, incidence):
    image_options = {
            'format': 'lammps-data',
            'atom_style': 'atomic',
            'specorder': ["Si", "O", "C", "H", "F"],
            }
    for image_idx in Image.save_snapshot_idx:
        image = Image.dump[image_idx]
        dst = f"incidences/{incidence}/{incidence}_{image_idx}"
        os.makedirs(dst, exist_ok=True)
        ase.io.write(f"{dst}/coo", image, **image_options)


def main():
    path_dir = sys.argv[1]

    with open("to_rlx.dat",'r') as O:
        save_list = [
            [int(image_idx) for image_idx in line.split()[1:]]
            for line in O.readlines()
        ]

    # nions = 50
    nions = 20
    for incidence in range(1, nions+1):
        src = f"{path_dir}/graph_{incidence}.pickle"
        with open(src, 'rb') as O:
            Image=pickle.load(O)
        Image.save_snapshot_idx = save_list[incidence]
        save_rlx_image(Image, incidence)


if __name__ == '__main__':
    main()
