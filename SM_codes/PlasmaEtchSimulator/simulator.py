"""
EtchSimulator: top-level orchestrator for iterative ion-bombardment MD.

    from PlasmaEtchSimulator import EtchSimulator
    EtchSimulator('input.yaml').run()
"""
import multiprocessing
import os
import shutil
import time

import ase

from .config import load_config
from .logger import Logger, NullLogger
from .params.bond_lengths import build_cutoff_matrix
from .calc.io import load_atom, make_convert_dict
from .calc.graph import get_builder
from .calc.byproduct import ByProductRemover
from .calc.postprocessor import PostProcessor
from .lmp.builder import LAMMPSInputBuilder
from .lmp.ions import write_ion_file
from .runner import get_runner


class EtchSimulator:
    """
    Orchestrates the full plasma-etch MD simulation loop.

    Parameters
    ----------
    config : str | dict
        Path to a YAML configuration file, or a pre-loaded config dict.
    """

    def __init__(self, config: 'str | dict') -> None:
        if isinstance(config, str):
            config = load_config(config)
        self.config = config

        self._setup_logger()
        self._print_config()
        self._setup_nprocs()
        self._setup_runner()

        self._base_dir = os.getcwd()

        self._cutoff_matrix = build_cutoff_matrix(config['pot_params']['elmlist'])
        self._graph_builder = get_builder(config['etc_params'].get('graph_backend', 'networkx'))
        self._bulk: 'ase.Atoms | None' = None
        self._byproduct_remover: 'ByProductRemover | None' = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Generate inputs and execute the full iterative simulation."""
        run_loc = self.config['etc_params']['run_loc']
        os.makedirs(run_loc, exist_ok=True)

        self._write_lammps_inputs(run_loc)
        self._write_ion_file(run_loc)
        self._load_shared_resources()

        nstart = self.config['run_params'].get('nstart', 1)
        nshoot = self.config['run_params']['nshoot']

        os.chdir(run_loc)
        try:
            self._initialize(nstart)
            for nit in range(nstart, nshoot + 1):
                self._run_iteration(nit)
        finally:
            os.chdir(self._base_dir)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_logger(self) -> None:
        log_cfg = self.config['etc_params'].get('log')
        run_loc = self.config['etc_params']['run_loc']
        os.makedirs(run_loc, exist_ok=True)
        if log_cfg and log_cfg != 'none':
            path = os.path.join(run_loc, 'log.txt') if log_cfg == 'auto' else log_cfg
            self.logger = Logger(path)
        else:
            self.logger = NullLogger()

    def _print_config(self) -> None:
        for section, params in self.config.items():
            self.logger.print(f'\n=== {section} ===')
            if isinstance(params, dict):
                for k, v in params.items():
                    self.logger.print(f'  {k:<30} : {v}')

    def _setup_nprocs(self) -> None:
        pottype = self.config['pot_params'].get('pottype', 'bpnn')
        if pottype in ('e3gnn', 'sevennet'):
            self.nprocs = 1
        else:
            nprocs_cfg = self.config['etc_params'].get('nprocs', 'auto')
            if nprocs_cfg == 'auto':
                self.nprocs = multiprocessing.cpu_count()
            else:
                self.nprocs = int(nprocs_cfg)
        self.logger.print(f'\nProcessors: {self.nprocs}')

    def _setup_runner(self) -> None:
        etc = self.config['etc_params']
        runner_type = etc.get('runner', 'subprocess')
        poll_interval = int(etc.get('poll_interval', 10))
        self.runner = get_runner(runner_type, poll_interval=poll_interval)

    def _write_lammps_inputs(self, run_loc: str) -> None:
        self._etch_input = os.path.join(run_loc, 'lammps.in')
        self._anneal_input = os.path.join(run_loc, 'lammps_anneal.in')
        LAMMPSInputBuilder.etch(self.config).write(self._etch_input)
        LAMMPSInputBuilder.anneal(self.config).write(self._anneal_input)
        self.logger.print(f'LAMMPS etch input  : {self._etch_input}')
        self.logger.print(f'LAMMPS anneal input: {self._anneal_input}')

    def _write_ion_file(self, run_loc: str) -> None:
        ion = self.config['etch_params']['ion']
        elmlist = self.config['pot_params']['elmlist']
        write_ion_file(os.path.join(run_loc, f'{ion}.dat'), ion, elmlist)

    def _load_shared_resources(self) -> None:
        rp = self.config['run_params']
        elmlist = self.config['pot_params']['elmlist']
        convert_dict = make_convert_dict(elmlist)
        self._bulk = load_atom(rp['bulk_loc'], convert_dict)
        self._byproduct_remover = ByProductRemover(
            elmlist=elmlist,
            byproduct_list=rp.get('byproduct_list', []),
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def _initialize(self, nstart: int) -> None:
        """Ensure the pre-iteration-0 structure exists and is post-processed."""
        n_prev = nstart - 1
        after_mod = f'str_shoot_{n_prev}_after_mod.coo'
        if os.path.exists(after_mod):
            return

        src = self.config['run_params'].get('slab_loc')
        raw_name = f'str_shoot_{n_prev}.coo'
        if src and not os.path.exists(raw_name):
            shutil.copy(src, raw_name)
        elif not os.path.exists(raw_name):
            raise FileNotFoundError(
                f"Initial slab not found at '{raw_name}'. "
                "Set 'run_params.slab_loc' in the config."
            )

        proc = self._make_processor(n_prev)
        proc.load()
        proc.build_graph()
        proc.remove_byproduct()
        proc.modify_slab()
        proc.save()

    def _run_iteration(self, nit: int) -> None:
        stime = time.time()
        iter_dir = self._iter_dir(nit)
        os.makedirs(iter_dir, exist_ok=True)

        proc = self._make_processor(nit)

        if proc.calc_done():
            self.logger.print(f'[{nit}] Already done; checking height update')
            self._maybe_update_incident_height(proc)
            return

        if not proc.has_input():
            self._run_lammps_etch(nit, iter_dir)

        proc.load()
        proc.build_graph()
        proc.remove_byproduct()

        anneal_needed = False
        if nit % self.config['run_params']['slab_update_iteration'] == 0:
            anneal_needed = proc.modify_slab()

        proc.save()
        self._maybe_update_incident_height(proc)

        elapsed = time.time() - stime
        self.logger.print(
            f'[{nit}] Etch done in {elapsed:.1f}s at {time.strftime("%H:%M:%S")}'
        )

        if anneal_needed:
            self._run_lammps_anneal(nit, iter_dir, proc)

    def _run_lammps_etch(self, nit: int, iter_dir: str) -> None:
        str_in = f'str_shoot_{nit - 1}_after_mod.coo'
        str_out = f'str_shoot_{nit}.coo'
        log_file = os.path.join(iter_dir, f'log_{nit}.lammps')
        cmd = self._build_cmd(self._etch_input, str_in, str_out, nit, log_name=log_file)
        self.logger.print(f'[{nit}] Running etch: {cmd}')
        if not self.runner.run(cmd, log_file=log_file):
            raise RuntimeError(f'LAMMPS etch failed at iteration {nit}')

    def _run_lammps_anneal(self, nit: int, iter_dir: str, proc: PostProcessor) -> None:
        str_before = f'str_shoot_{nit}_after_mod_before_anneal.coo'
        str_after = f'str_shoot_{nit}_after_mod.coo'
        if os.path.exists(str_after):
            shutil.move(str_after, str_before)
        log_file = os.path.join(iter_dir, f'log_{nit}_anneal.lammps')
        cmd = self._build_cmd(
            self._anneal_input, str_before, str_after, nit,
            log_name=log_file, suffix='_anneal',
        )
        self.logger.print(f'[{nit}] Running anneal: {cmd}')
        if not self.runner.run(cmd, log_file=log_file):
            raise RuntimeError(f'LAMMPS anneal failed at iteration {nit}')

    # ------------------------------------------------------------------
    # LAMMPS command construction
    # ------------------------------------------------------------------

    def _build_cmd(self, input_file: str, strin: str, strout: str, nit: int,
                   log_name: str = '', suffix: str = '') -> str:
        lmp = self.config['etc_params']['lmp_loc']
        stdout = f'stdout_{nit}{suffix}.x'
        cmd = (
            f'{lmp}'
            f' -in {input_file}'
            f' -v strin {strin}'
            f' -v strout {strout}'
            f' -v i {nit}'
            f' -l {log_name or f"log_{nit}{suffix}.lammps"}'
            f' >& {stdout}'
        )
        if self.nprocs > 1:
            cmd = f'mpirun -np {self.nprocs} ' + cmd
        return cmd

    # ------------------------------------------------------------------
    # Incident height update
    # ------------------------------------------------------------------

    def _maybe_update_incident_height(self, proc: PostProcessor) -> None:
        """Raise incident_height and evapheight when the slab grows."""
        if self.config['run_params']['slab_modify_crit'] != 'pen_depth':
            return
        if not os.path.exists('add_' + proc.name_in):
            return

        h = proc.slab_height()
        if h is None:
            return
        if self.config['etch_params']['incident_height'] > h * 1.5:
            return

        delta = self.config['run_params']['h_add_amount']
        self.config['etch_params']['incident_height'] += delta
        self.config['etch_params']['evapheight'] += delta
        self.logger.print(
            f'Incident height updated to {self.config["etch_params"]["incident_height"]:.2f} Å'
        )
        LAMMPSInputBuilder.etch(self.config).write(self._etch_input)

    # ------------------------------------------------------------------
    # Factory / helpers
    # ------------------------------------------------------------------

    def _make_processor(self, iteration: int) -> PostProcessor:
        return PostProcessor(
            iteration=iteration,
            config=self.config,
            bulk=self._bulk,
            cutoff_matrix=self._cutoff_matrix,
            byproduct_remover=self._byproduct_remover,
            graph_builder=self._graph_builder,
            logger=self.logger,
        )

    @staticmethod
    def _iter_dir(nit: int) -> str:
        return f'iter_{nit:05d}'
