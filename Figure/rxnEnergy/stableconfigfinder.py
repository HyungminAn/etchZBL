import os
import sys
import numpy as np
from ase.io import read, write

from utils import PARAMS

class StableConfigFinder:
    def run(self, src):
        line = "#incidence,snapshots_idx\n"
        imgs_to_save = {}
        for incidence in range(1, PARAMS.DUMP_INFO.n_incidences+1):
            path_dump = f"{src}/dump_{incidence}.lammps"
            dump = read(path_dump, format='lammps-dump-text', index=':')
            stable_configs = self.get_stable_config(dump)
            line += f"{incidence} " + " ".join(map(str, stable_configs)) + "\n"
            for image_idx in stable_configs:
                if incidence not in imgs_to_save:
                    imgs_to_save[incidence] = []
                imgs_to_save[incidence].append((image_idx, dump[image_idx]))

        self.save(imgs_to_save)

        with open(PARAMS.path_to_rlx, 'w') as rlx_fo:
            rlx_fo.write(line)

    def get_stable_config(self, dump):
        '''
        Just extract the 10% and 60% of the total number of images
        '''
        n_images = len(dump)
        result = [int(np.round(n_images * 0.1)),
                  int(np.round(n_images * 0.6))]
        return result

    def save(self, imgs_to_save):
        for incidence, images in imgs_to_save.items():
            for image_idx, image in images:
                dst = f"{PARAMS.path_incidences}/{incidence}/{incidence}_{image_idx}"
                os.makedirs(dst, exist_ok=True)
                write(f"{dst}/coo", image, **PARAMS.SYSTEM_DEPENDENT.LAMMPS_SAVE_OPTS)

def main():
    if len(sys.argv) != 2:
        print("Usage: python stableconfigfinder.py <path_src>")
        sys.exit(1)

    scf = StableConfigFinder()
    scf.run(sys.argv[1])

if __name__ == '__main__':
    main()
