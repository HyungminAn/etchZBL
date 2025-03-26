import os
import copy

import ase
import ase.io
import ase.build


class FileNameSetter:
    def __init__(self):
        pass

    @staticmethod
    def set_name_from_protocol(protocol):
        fname_in, fname_out = None, None
        if protocol == 'default':
            fname_in = FileNameSetter.default_fname_in
            fname_out = FileNameSetter.default_fname_out
        elif protocol == 'custom':
            fname_in = FileNameSetter.custom_fname_in
            fname_out = FileNameSetter.custom_fname_out
        return fname_in, fname_out

    @staticmethod
    def default_fname_in(it) -> str:
        return f'str_shoot_{it}.coo'
    @staticmethod
    def default_fname_out(it) -> str:
        return f'str_shoot_{it}_after_mod.coo'

    @staticmethod
    def custom_fname_in(it) -> str:
        return f'HF_shoot_{it}.coo'
    @staticmethod
    def custom_fname_out(it) -> str:
        return f'HF_shoot_{it}_after_removing.coo'


def save_atom(calc, fname : str, atoms : ase.Atoms) -> None:
    save_options = {
            'format': 'lammps-data',
            'specorder': calc.elmlist,
            'velocities': True,
            }
    try:
        ase.io.write(fname, atoms, **save_options)
    except Exception as _:
        save_options.pop('velocities')
        ase.io.write(fname, atoms, **save_options)
    assert os.path.exists(fname), f'Error in writing {fname}'
    return

def load_atom(fname : str, convert_dict) -> ase.Atoms:
    '''
    The function that set input structure name and output structure name

    atoms.cell is 3x3 matrix, positions is Nx3 matrix and  Only cubic cell is supported
    '''
    atoms = ase.io.read(fname, format='lammps-data', sort_by_id = False)
    chem_sym = set(atoms.get_chemical_symbols())
    if chem_sym.issubset(convert_dict.keys()):
        for atom in atoms:
            atom.symbol = convert_dict[atom.symbol]

    cell = atoms.cell
    positions = atoms.positions
    for i in range(3):
        positions[:,i] = positions[:,i] % cell[i,i]
    atoms.positions = positions
    return atoms

def init_out(box_height, atoms : ase.Atoms) -> None:
    '''
    Fix all positions to the PBC cell only change the Z box slab size
    Set the all positions of atom in PBC cell
    '''
    oatoms = copy.deepcopy(atoms)
    positions = oatoms.positions
    for i in range(3):
        positions[:,i] = positions[:,i] % oatoms.cell[i,i]

    oatoms.cell[2,2] = box_height
    oatoms.positions = positions
    return oatoms
