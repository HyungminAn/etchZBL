import os # import copy

# import numpy as np
import torch
import networkx as nx

import ase
import ase.io
import ase.build
# from ase.data import chemical_symbols , covalent_radii
from ase.data import chemical_symbols

from PlasmaEtchSimulator.calc.byproduct import ByProductRemover
from PlasmaEtchSimulator.calc.functions import FileNameSetter
from PlasmaEtchSimulator.calc.functions import save_atom
from PlasmaEtchSimulator.calc.functions import load_atom
from PlasmaEtchSimulator.calc.slabmodifier import SlabModifier


class DistMatrix:
    def __init__(self):
        self.matrix = self.gen_matrix()

    def gen_matrix(self):
        bond_length = torch.tensor([
            [2.96712, 1.98784, 2.24319, 2.00469, 2.12597],
            [0.00000, 1.60294, 1.48655, 1.28316, 1.78380],
            [0.00000, 0.00000, 1.70875, 1.47971, 1.68691],
            [0.00000, 0.00000, 0.00000, 0.97536, 1.21950],
            [0.00000, 0.00000, 0.00000, 0.00000, 1.85160],
        ])
        return bond_length + bond_length.t() - torch.diag(bond_length.diag())


class StrProcess:
    device           = None
    distance_matrix  = DistMatrix().matrix
    elmlist          = None
    logger           = None
    atom_bulk        : ase.Atoms = None
    etch_params      : dict = {}
    byproduct        : 'ByProduct' = None
    KEY              = None
    fix              : float = None

    def __init__(self,
                iteration : int,
                run_params : dict,
                etc_params : dict,
                elmlist : list,
                log = None,
                KEY = None,
                _naming = 'default') -> None:
        """This object save the information of each etching simulation
            such as the height of the slab,
            the temperature of the slab,
            the molecule to be incident,
            and calculation is done etc...
        """
        self.iteration = iteration
        self.run_params = run_params
        for key, value in run_params.items():
            setattr(self, key, value)
        self.etc_params = etc_params
        for key, value in etc_params.items():
            setattr(self, key, value)

        self.bulk_loc : str
        self.nshoot     : int = 1

        self.slab_max_height : float
        self.slab_min_height : float
        self.slab_center_height : float
        self.slab_update_iteration : int
        self.box_height            : float
        if iteration % self.slab_update_iteration == 0:
            self.check_slab = True
        else:
            self.check_slab = False
        self.slab_height : float    = None

        self.fname_in, self.fname_out = \
            FileNameSetter.set_name_from_protocol(_naming)
        self.name_in    : str       = self.fname_in(iteration)
        self.name_out   : str       = self.fname_out(iteration)
        self.exists_in  = os.path.exists(self.name_in)
        self.exists_out = os.path.exists(self.name_out)

        self.str_in     : ase.Atoms = None
        self.str_out    : ase.Atoms = None
        self.cmd        : str       = None
        self.convert_dict = {chemical_symbols[idx+1]: symbol  for idx, symbol in enumerate(elmlist)}

        self._set_class_variable(KEY,
                                 self.etch_params,
                                 self.run_params,
                                 etc_params,
                                 elmlist,
                                 log)

        self.G_in           : nx.Graph  = None
        self.cluster_in     : list = None
        self.G_out          : nx.Graph  = None
        self.cluster_out    : list = None
        self.byproduct_removed : bool = False

    def update(self):
        self.exists_in = os.path.exists(self.name_in)
        self.exists_out = os.path.exists(self.name_out)
        try:
            if self.exists_in and self.str_in == None:
                self.str_in =  load_atom(self.name_in, self.convert_dict)
            if self.exists_out and self.str_out == None:
                self.str_out = load_atom(self.name_out, self.convert_dict)
        except Exception as e:
            self.logger.print(f'Error occured during loading {self.name_in} : {e}')
            self.logger.print(f'Terminating simulation')
            exit()

    def save(self) -> None:
        """Save the information of the calculation
        """
        if self.calc_done():
            self.logger.print(f'{self.name_out} already exists')
            return
        save_atom(self, self.name_out, self.str_out)
        self.logger.print(f'{self.name_out} written')

    def calc_done(self)-> bool:
        return self.exists_out

    def has_input(self) -> bool:
        return self.exists_in

    @classmethod
    def set_etch_params(cls, etch_params : dict) -> None:
        cls.etch_params = etch_params
        cls.fix = etch_params['fix']

    def remove_byproduct(self):
        runner = self.byproduct_remover
        runner.run(self)

    def slab_modify_height(self):
        runner = SlabModifier()
        runner.run(self)

    def _set_class_variable(self,
                            KEY,
                            etch_params,
                            run_params,
                            etc_params,
                            elmlist,
                            log):
        ## If key is empty, load KEY from the etchpy.key
        if KEY == None:
            from PlasmaEtchSimulator.key import KEY
        if StrProcess.KEY == None:
            StrProcess.KEY = KEY

        if StrProcess.device == None:
            ## Set the device to be used, if the 'device' is set to 'gpu', the cuda device is used if possbile
            if etc_params.get(KEY.DEVICE) == 'gpu':
                StrProcess.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            elif etch_params.get(KEY.DEVICE) == 'cpu':
                StrProcess.device = 'cpu'

        if StrProcess.elmlist == None:
            StrProcess.elmlist    : list = elmlist
        if StrProcess.logger == None:
            StrProcess.logger = log
        if StrProcess.atom_bulk == None:
            StrProcess.atom_bulk = load_atom(self.bulk_loc, self.convert_dict)
        if StrProcess.byproduct == None:
            StrProcess.byproduct_remover = ByProductRemover(elmlist = self.elmlist,
                                                            byproduct = run_params['byproduct_list'],
                                                            logger=self.logger)
