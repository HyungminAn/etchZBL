import os
from ase.io import read, write


def main():
    src = "../dft"
    for root, dirs, files in os.walk(src):
        for file in files:
            if not file == "POSCAR":
                continue

            path = os.path.join(root, file)
            dst = os.path.basename(os.path.dirname(path))
            if os.path.exists(f"{dst}/input.data"):
                continue

            os.makedirs(dst, exist_ok=True)
            atoms = read(path, format="vasp")
            write(f"{dst}/input.data", atoms, format="lammps-data", specorder=["Si", "O", "C", "H", "F"])
            print(f"Generated {dst}/input.data")


if __name__ == "__main__":
    main()
