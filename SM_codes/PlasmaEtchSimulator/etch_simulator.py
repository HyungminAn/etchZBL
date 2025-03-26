#!/bin/python
import os
import shutil
import time
import subprocess

from PlasmaEtchSimulator.lmp.etch import LAMMPSetch
from PlasmaEtchSimulator.lmp.anneal import LAMMPSanneal
# from PlasmaEtchSimulator.lmp.anneal import LAMMPSanneal
from PlasmaEtchSimulator.lmp.functions import ion_writer
from PlasmaEtchSimulator.calc.strprocess import StrProcess
from PlasmaEtchSimulator.key import KEY
"""
LAMMPS simulcation codes uses etch_calc, lammps_etch
Written by Sangmin Oh (2024.11.22)
"""


class Logger:
    """Logger class which save information
    """
    def __init__(self, location) -> None:
        if os.path.exists(location):
            src = location
            count = 1
            while os.path.exists(f'{location}_{count}'):
                count += 1
            dst = f'{location}_{count}'
            os.rename(src, dst)
        self.fileobj = open(location, 'w', buffering=1)
    def print(self, line : str) -> None:
        self.fileobj.write(line+'\n')
        print(line)
    def log(self, line : str) -> None:
        self.fileobj.write(line)


class EtchSimulator:
    def __init__(self,
                pot_params : dict,
                etch_params : dict,
                run_params : dict,
                etc_params : dict) -> None:
        """Initilize the LAMMPSetch class

        Args:
            pot_params (dict): parameters related to potential definition
            etch_params (dict): parameters related to etching simulation like molecule, slab, energy, angle ...
            etc_params (dict): parameters related to etc like log file, dump file, ...
        """
        self.pot_params     = pot_params
        self.etch_params    = etch_params
        self.run_params     = run_params
        self.etc_params     = etc_params

        self._set_log()
        self._print_information()

        self.lmp_etch = LAMMPSetch(pot_params,
                                   etch_params,
                                   etc_params,
                                   logger=self.log)
        self.lmp_anneal = LAMMPSanneal(pot_params,
                                       etch_params,
                                       etc_params,
                                       logger=self.log)
        self.lmp_calc = []
        self.base_loc = os.getcwd()

        self._set_nprocs()

    def init(self):
        """Initilize the etching simulation : create lammps.in script, molecule file, ...
        """
        self.lmp_etch.generate_lammps_input(lmpdir=self.etc_params[KEY.RUN_LOC])
        self.lmp_anneal.generate_lammps_input(lmpdir=self.etc_params[KEY.RUN_LOC])
        ion_writer(os.path.join(self.etc_params[KEY.RUN_LOC],
                                f'{self.etch_params[KEY.ION]}.dat'),
                   self.etch_params[KEY.ION],
                   self.pot_params[KEY.ELMLIST])

    def run(self):
        """Run the etching simulation
        First create slab and relax it, then run the etching simulation
        Check the slab generattion is done by filename
        Copy the slab data from the given location
        """
        StrProcess.set_etch_params(self.etch_params)
        os.chdir(self.etc_params[KEY.RUN_LOC])

        nstart = self.run_params[KEY.NSTART] if KEY.NSTART in self.run_params.keys() else 1
        self._initialize_run(nstart)

        nshoot = self.run_params[KEY.NSHOOT]
        for nit in range(nstart,nshoot+1):
            self._run_one_incidence(nit)

        os.chdir(self.base_loc)

    def _set_log(self):
        '''
        Set the log file location
        '''
        os.makedirs(self.etc_params[KEY.RUN_LOC], exist_ok=True)
        if KEY.LOG in self.etc_params.keys():
            logloc = self.etc_params[KEY.LOG]
            if logloc == 'auto':
                logloc = os.path.join(self.etc_params[KEY.RUN_LOC],
                                      'log.txt')
            self.log = Logger(logloc)
        else:
            self.log = None

    def _print_information(self):
        '''
        Show all information to log
        '''
        pot_params = self.pot_params
        etch_params = self.etch_params
        run_params = self.run_params
        etc_params = self.etc_params

        self.log.print('Potential Parameters')
        for key, value in pot_params.items():
            self.log.print(f'{key:20} : {value}')
        self.log.print('Etching Parameters')
        for key, value in etch_params.items():
            self.log.print(f'{key:20} : {value}')
        self.log.print('Run Parameters')
        for key, value in run_params.items():
            self.log.print(f'{key:20} : {value}')
        self.log.print('Etc Parameters')
        for key, value in etc_params.items():
            self.log.print(f'{key:20} : {value}')

    def _set_nprocs(self):
        '''
        Get the number of processors automatically
        '''
        if (self.pot_params[KEY.POT_TYPE] == 'e3gnn'):
            self.nprocs = 1
        elif KEY.NPROCS in self.etc_params.keys() and self.etc_params[KEY.NPROCS] == 'auto':
            import multiprocessing
            self.nprocs = multiprocessing.cpu_count()
        elif KEY.NPROCS in self.etc_params.keys():
            self.nprocs = self.etc_params[KEY.NPROCS]
            self.nprocs = int(self.nprocs)
        self.log.print(f'Number of processors : {self.nprocs}')

    def _get_cmd(self,
                input : str,
                strin : str,
                strout : str,
                it : int) -> str:
        """Get the command to run the etching simulation
        """
        cmd = (
                f' {self.etc_params[KEY.LMP_LOC]}'
                f' -in {input}'
                f' -v strin {strin}'
                f' -v strout {strout}'
                f' -v i {it}'
                f' -l log_{it}.lammps >& stdout_{it}.x'
                )
        if self.nprocs > 1:
            cmd = f'mpirun -np {self.nprocs}' + cmd
        return cmd

    def _get_cmd_anneal(self,
                input : str,
                strin : str,
                strout : str,
                it : int) -> str:
        """Get the command to run the etching simulation
        """
        cmd = (
                f' {self.etc_params[KEY.LMP_LOC]}'
                f' -in {input}'
                f' -v strin {strin}'
                f' -v strout {strout}'
                f' -v i {it}'
                f' -l log_{it}_anneal.lammps >& stdout_{it}_anneal.x'
                )
        if self.nprocs > 1:
            cmd = f'mpirun -np {self.nprocs}' + cmd
        return cmd

    def _initialize_run(self, nstart):
        '''
        Initialize the etching simulation
        '''
        n_prev = nstart-1
        path_str_after_mod = f'str_shoot_{n_prev}_after_mod.coo'
        if os.path.exists(path_str_after_mod):
            return

        # if KEY.SLAB_LOC in self.run_params.keys():
        #     path_initial_slab = self.run_params[KEY.SLAB_LOC]
        #     os.system(f'cp {path_initial_slab} str_shoot_0.coo')

        path_str_needed = f'str_shoot_{n_prev}.coo'
        if not os.path.exists(path_str_needed):
            print(f'Initial slab is not found at {path_str_needed}')
            # self.lmp_init.initial_run(lmpdir=self.etc_params[KEY.RUN_LOC])
            # cmd = self._get_cmd(self.lmp_init.get_lammps_input(), self.lmp_init.slabname, 'str_shoot_0.coo', 0)
            # self.log.print(cmd)
            # cal_ok = subprocess.run(cmd, shell=True)

        calc = StrProcess(n_prev,
                          self.run_params,
                          self.etc_params,
                          elmlist=self.pot_params[KEY.ELMLIST],
                          log = self.log)
        calc.update()
        calc.remove_byproduct()
        calc.slab_modify_height()
        calc.save()

    def _run_one_incidence(self, nit):
        stime = time.time()
        calc = StrProcess(nit,
                          self.run_params,
                          self.etc_params,
                          elmlist=self.pot_params[KEY.ELMLIST],
                          log=self.log)
        to_check_slab_height = nit % self.run_params[KEY.SLAB_UPDATE_ITERATION] == 0
        if calc.calc_done():
            self.log.print(f'Calculation is already done at {nit} iteration')
            is_lmp_input_updated = calc.update_height_parameter()
            if is_lmp_input_updated:
                self.lmp_etch._update_self()
                self.lmp_etch.generate_lammps_input(lmpdir=self.etc_params[KEY.RUN_LOC])
            return

        if not calc.has_input():
            str_in = f'str_shoot_{nit-1}_after_mod.coo'
            str_out = f'str_shoot_{nit}.coo'
            cmd = self._get_cmd(self.lmp_etch.get_lammps_input(),
                               str_in,
                               str_out,
                               nit)

            self.log.print(f'Run {nit} iteration simulation at {time.strftime("%Y-%m-%d %H:%M:%S")}')

            cal_ok = subprocess.run(cmd, shell=True)
            assert cal_ok.returncode == 0, f'Error in {nit} iteration simulation'

            time_spent = time.time()-stime
            self.log.print(f'Run {nit} iteration simulation at {time.strftime("%Y-%m-%d %H:%M:%S")} : time spent {time_spent:.2f} sec')

        calc.update()
        calc.remove_byproduct()
        if to_check_slab_height:
            is_anneal_needed = calc.slab_modify_height()

        calc.save()
        is_lmp_input_updated = calc.update_height_parameter()
        if is_lmp_input_updated:
            self.lmp_etch._update_self()
            self.lmp_etch.generate_lammps_input(lmpdir=self.etc_params[KEY.RUN_LOC])

        self.lmp_calc.append(calc)

        if to_check_slab_height and is_anneal_needed:
            str_in = f'str_shoot_{nit}_after_mod_before_anneal.coo'
            str_out = f'str_shoot_{nit}_after_mod.coo'
            shutil.move(str_out, str_in)
            cmd = self._get_cmd_anneal(self.lmp_anneal.get_lammps_input(),
                                       str_in,
                                       str_out,
                                       nit)

            self.log.print(f'Run {nit} iteration anneal at {time.strftime("%Y-%m-%d %H:%M:%S")}')

            cal_ok = subprocess.run(cmd, shell=True)
            assert cal_ok.returncode == 0, f'Error in {nit} iteration anneal'

            time_spent = time.time()-stime
            self.log.print(f'Run {nit} iteration anneal at {time.strftime("%Y-%m-%d %H:%M:%S")} : time spent {time_spent:.2f} sec')
