import os
import torch
import ase
from ase.data import chemical_symbols
from PlasmaEtchSimulator.calc.byproduct import ByProductRemover
from PlasmaEtchSimulator.calc.functions import FileNameSetter, save_atom, load_atom
from PlasmaEtchSimulator.calc.slabmodifier_height import SlabModifier as SlabMod_h
from PlasmaEtchSimulator.calc.slabmodifier_pendepth import SlabModifier as SlabMod_pd
from PlasmaEtchSimulator.key import KEY as DEFAULT_KEY
from PlasmaEtchSimulator.params import DistMatrix


class StrProcess:
    _is_initialized = False
    device = None
    distance_matrix = None
    elmlist = None
    # logger = None
    atom_bulk: ase.Atoms = None
    etch_params: dict = {}
    byproduct_remover = None
    KEY = None
    fix: float = None

    def __init__(self,
                 iteration,
                 run_params,
                 etc_params,
                 elmlist,
                 log=None,
                 KEY=None,
                 _naming='default'):
        self.iteration = iteration
        self.run_params = run_params
        self.etc_params = etc_params
        self._naming = _naming
        self.logger = log

        StrProcess._initialize_class_variables(etc_params,
                                               elmlist,
                                               log,
                                               KEY,
                                               run_params)
        self._initialize_instance_variables()

    @classmethod
    def _initialize_class_variables(cls,
                                    etc_params,
                                    elmlist,
                                    log,
                                    KEY,
                                    run_params):
        if not cls._is_initialized:
            cls.KEY = KEY if KEY is not None else DEFAULT_KEY
            cls.distance_matrix = DistMatrix().matrix
            cls._set_device(etc_params)
            cls.elmlist = elmlist
            # cls.logger = log
            cls.atom_bulk = load_atom(run_params[cls.KEY.BULK_LOC],
                                      cls._create_convert_dict(elmlist))
            cls.byproduct_remover = ByProductRemover(elmlist=elmlist,
                                                     byproduct=run_params['byproduct_list'],
                                                     logger=log)
            cls._is_initialized = True

    @classmethod
    def _set_device(cls, etc_params):
        if cls.KEY.DEVICE in etc_params:
            if etc_params[cls.KEY.DEVICE] == 'gpu':
                cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            elif etc_params[cls.KEY.DEVICE] == 'cpu':
                cls.device = 'cpu'

    @staticmethod
    def _create_convert_dict(elmlist):
        return {chemical_symbols[idx+1]: symbol for idx, symbol in enumerate(elmlist)}

    def _initialize_instance_variables(self):
        for key, value in self.run_params.items():
            setattr(self, key, value)
        for key, value in self.etc_params.items():
            setattr(self, key, value)

        self.check_slab = self.iteration % self.slab_update_iteration == 0
        self.fname_in, self.fname_out = FileNameSetter.set_name_from_protocol(self._naming)
        self.name_in = self.fname_in(self.iteration)
        self.name_out = self.fname_out(self.iteration)
        self.exists_in = os.path.exists(self.name_in)
        self.exists_out = os.path.exists(self.name_out)
        self.convert_dict = self._create_convert_dict(self.elmlist)

        self.G_in = None
        self.cluster_in = None
        self.G_out = None
        self.cluster_out = None
        self.byproduct_removed = False

        self.str_in = None
        self.str_out = None

    @classmethod
    def set_etch_params(cls, etch_params: dict) -> None:
        cls.etch_params = etch_params
        cls.fix = etch_params['fix']

    def update(self):
        self.exists_in = os.path.exists(self.name_in)
        self.exists_out = os.path.exists(self.name_out)
        try:
            if self.exists_in and self.str_in is None:
                self.str_in = load_atom(self.name_in, self.convert_dict)
            if self.exists_out and self.str_out is None:
                self.str_out = load_atom(self.name_out, self.convert_dict)
        except Exception as e:
            self.logger.print(f'Error occurred during loading {self.name_in} : {e}')
            self.logger.print(f'Terminating simulation')
            exit()

    def save(self) -> None:
        if self.calc_done():
            self.logger.print(f'{self.name_out} already exists')
            return
        save_atom(self, self.name_out, self.str_out)
        self.logger.print(f'{self.name_out} written')

    def calc_done(self) -> bool:
        return self.exists_out

    def skip_calc(self, crit) -> bool:
        if self.iteration < crit:
            return True
        return False

    def has_input(self) -> bool:
        return self.exists_in

    def remove_byproduct(self):
        runner = self.byproduct_remover
        runner.run(self)

    def slab_modify_height(self):
        if self.run_params['slab_modify_crit'] == 'pen_depth':
            runner = SlabMod_pd()
        elif self.run_params['slab_modify_crit'] == 'height':
            runner = SlabMod_h()
        else:
            raise ValueError('Invalid slab_modify_crit: only "pen_depth" or "height"')
        is_anneal_needed = runner.run(self)
        return is_anneal_needed

    def update_height_parameter(self):
        '''
        If the height of slab has changed,
        the corresonding variables should be updated also.
        '''
        if not self.run_params['slab_modify_crit'] == 'pen_depth':
            return False

        path_add = f'add_{self.name_in}'
        if not os.path.exists(path_add):
            return False

        runner = SlabMod_pd()
        self.update()
        slab_z = runner.get_slab_height(self, ctype='out')
        is_vacuum_too_large = self.etch_params['incident_height'] > slab_z * 1.5
        if is_vacuum_too_large:
            return False

        h_change_amount = self.h_add_amount
        self.etch_params['incident_height'] += h_change_amount
        self.etch_params['evapheight'] += h_change_amount
        return True
