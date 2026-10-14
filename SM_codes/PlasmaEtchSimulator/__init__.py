"""
PlasmaEtchSimulator: iterative ion-bombardment MD workflow with LAMMPS.
"""
from .simulator import EtchSimulator
from .config import load_config

__all__ = ['EtchSimulator', 'load_config']
__version__ = '0.2.0'
