import os
from pymatgen.io.vasp import Poscar
import shutil as shu
import numpy as np


pm_structure = Poscar.from_file("rlx/CONTCAR").structure

os.makedirs("eos/0", exist_ok=True)
shu.copy("rlx/CONTCAR", "eos/0/POSCAR")

z_atom_1 = pm_structure.cart_coords[0, 2]
z_atom_2 = pm_structure.cart_coords[1, 2]
bond_length = np.abs(z_atom_1 - z_atom_2)

pm_structure = Poscar.from_file("rlx/CONTCAR").structure
N = 0
while N <= 30:
    N += 1
    os.makedirs(f"eos/{N}", exist_ok=True)
    pm_structure.translate_sites(1, [0, 0, 0.05], frac_coords=False)

    Poscar(structure=pm_structure, comment="edited").write_file(f'eos/{N}/POSCAR')
    print(f'{N} written')

pm_structure = Poscar.from_file("rlx/CONTCAR").structure
N = 0
while N <= 10:
    N += 1
    os.makedirs(f"eos/_{N}", exist_ok=True)
    pm_structure.translate_sites(1, [0, 0, -0.05], frac_coords=False)

    Poscar(structure=pm_structure, comment="edited").write_file(f'eos/_{N}/POSCAR')
    print(f'_{N} written')
