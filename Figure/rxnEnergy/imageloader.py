import os
from ase.io import read, write
import pickle

from utils import PARAMS

class ImageLoader:
    def run(self):
        with open(PARAMS.path_to_rlx, 'r') as f:
            lines = f.readlines()[1:]
        coos = []

        for line in lines:
            incidence, *img_indices = line.strip().split()
            for img_idx in img_indices:
                src = f"{PARAMS.path_incidences}/{incidence}/{incidence}_{img_idx}/rlx.coo"
                if not os.path.exists(src):
                    print(f"File {src} does not exist, please finish the relaxation first.")
                    return None
                coos.append(read(src, **PARAMS.LAMMPS_READ_OPTS))

        with open(PARAMS.path_nnp_pickle, 'wb') as f:
            pickle.dump(coos, f)

        write(PARAMS.path_nnp_extxyz, coos, format='extxyz')
        print(f'Saved {len(coos)} images to {PARAMS.path_nnp_extxyz}')

        return coos


if __name__ == '__main__':
    il = ImageLoader()
    il.run()
