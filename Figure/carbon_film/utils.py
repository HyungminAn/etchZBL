import time
from functools import wraps
from dataclasses import dataclass

from ase.io import read


def timeit(function):
    '''
    Wrapper function to measure the execution time of a function.
    '''
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print(f'{function.__name__:40s} took {end - start:10.4f} seconds')
        return result
    return wrapper


@dataclass
class READ_OPTS:
    ATOM_NUM_SI = 14
    ATOM_NUM_O = 8
    ATOM_NUM_C = 6
    ATOM_NUM_H = 1
    ATOM_NUM_F = 9

    ATOM_IDX_SI = 1
    ATOM_IDX_O = 2
    ATOM_IDX_C = 3
    ATOM_IDX_H = 4
    ATOM_IDX_F = 5

    LAMMPS_DATA = {
            'format': 'lammps-data',
            'Z_of_type': {
                ATOM_IDX_SI: ATOM_NUM_SI,
                ATOM_IDX_O: ATOM_NUM_O,
                ATOM_IDX_C: ATOM_NUM_C,
                ATOM_IDX_H: ATOM_NUM_H,
                ATOM_IDX_F: ATOM_NUM_F
                },
            'sort_by_id': False,
            }

def read_structure(path_to_input):
    """
    Read the atomic structure from a LAMMPS data file using ASE.
    We specify Z_of_type to map LAMMPS type -> Z (atomic number),
    which in turn sets .symbol to [Si, O, C, H, F].
    """
    # Example mapping for a system with 5 types (Si, O, C, H, F).
    # Adjust or remove if your data is different.
    atoms = read(path_to_input, **READ_OPTS.LAMMPS_DATA)
    return atoms
