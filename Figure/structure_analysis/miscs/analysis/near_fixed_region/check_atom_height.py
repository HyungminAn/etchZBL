import os
import numpy as np

from ase.io import read


def main():
    src = "/data2_1/andynn/Etch/05_EtchingMD/ver2_2/02_CF3/50eV_LargeCell"
    slab_atoms = ["Si", "O"]
    h_fix = 6  # Angstrom
    read_coo_options = {
            'format': 'lammps-data',
            'Z_of_type': {
                1: 14,
                2: 8,
                3: 6,
                4: 1,
                5: 9,
                }
            }

    file_list = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith("_after_mod.coo"):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    file_list = sorted(file_list, key=lambda x: int(x.split("/")[-1].split("_")[2]))

    for file in file_list:
        atoms = read(file, **read_coo_options)

        height_list = np.array([atom.position[2] for atom in atoms if atom.symbol not in slab_atoms])
        if len(height_list) == 0:
            print(f"{file}: no atoms found ;; continue")
            continue

        h_min = np.min(height_list)

        if h_min < h_fix:
            print(f"{file}: {h_min} < {h_fix} ;; break")
            break
        else:
            print(f"{file}: {h_min} > {h_fix} ;; continue")


if __name__ == "__main__":
    main()
