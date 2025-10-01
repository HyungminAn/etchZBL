import numpy as np
import os
from pymatgen.io.vasp import Poscar
import pymatgen.analysis.elasticity as elast
rlx_structure = Poscar.from_file("CONTCAR").structure

strains = np.arange(-10, 0) * 0.01
strains = np.concatenate((strains, [0], -1*np.flip(strains))).tolist()
Deform = elast.DeformedStructureSet(
        rlx_structure, norm_strains=strains, shear_strains=strains,
        symmetry=True
        )

os.makedirs("structures", exist_ok=True)

for e, i in enumerate(Deform):
    os.makedirs(f"structures/{e}", exist_ok=True)
    to_save_poscar = Poscar(structure=i, comment="edited")
    to_save_poscar.write_file(f"structures/{e}/POSCAR")
