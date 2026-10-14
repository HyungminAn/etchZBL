"""
Penetration-depth-based slab modification strategy.

Decisions are driven by where C/H/F atoms are located relative to thresholds,
not by absolute slab height.
"""
import os
import shutil

import numpy as np

from ..io import save_atom, load_atom
from ._core import expand_slab, expand_slab_from_saved, trim_slab


_CHF = {'C', 'H', 'F'}


class PenDepthModifier:
    """
    Trim when the surface is CHF-rich (etched material piling up).
    Expand when CHF ions have penetrated below the threshold depth.
    """

    def run(self, proc) -> bool:
        """Modify proc.str_out based on CHF penetration depth.
        Returns True when annealing is required."""
        if proc.calc_done():
            return False

        rp = proc.config['run_params']
        ep = proc.config['etch_params']
        n_limit = rp['n_atoms_limit']
        h_CHF_crit = rp['h_CHF_crit']
        CHF_ratio_crit = rp['CHF_ratio_crit']
        h_pen_crit = rp['h_penetrate_crit']
        h_sub = rp['h_sub_amount']
        h_add = rp['h_add_amount']
        fix = ep['fix']

        saved_path = self._prev_saved_path(proc)
        saved = load_atom(saved_path, proc.convert_dict) if os.path.exists(saved_path) else None
        n_atoms = len(proc.str_in)

        chf_ratio = self._chf_ratio_above(proc.str_in, h_CHF_crit)

        if chf_ratio > CHF_ratio_crit:
            if n_atoms > n_limit:
                proc.logger.print(
                    f"CHF ratio {chf_ratio:.3f} > {CHF_ratio_crit:.3f} "
                    f"and n_atoms {n_atoms} > {n_limit}; trimming {h_sub:.2f} Å"
                )
                proc.str_out, archive = trim_slab(proc.str_out, h_sub, saved)
                save_atom(proc.str_out, 'sub_' + proc.name_in, proc.elmlist)
                save_atom(archive, 'save_' + proc.name_in, proc.elmlist)
            else:
                proc.logger.print(
                    f"CHF ratio {chf_ratio:.3f} > {CHF_ratio_crit:.3f} "
                    f"but n_atoms {n_atoms} <= {n_limit}; skipping trim"
                )
            return False

        if self._is_penetrated(proc.str_in, h_pen_crit):
            if n_atoms > n_limit:
                raise RuntimeError(
                    f"Slab penetrated but n_atoms ({n_atoms}) exceeds limit ({n_limit}). "
                    "Adjust n_atoms_limit or h_penetrate_crit in the config."
                )
            proc.logger.print(
                f"CHF atoms below {h_pen_crit:.2f} Å threshold; expanding by {h_add:.2f} Å"
            )
            if saved is not None:
                proc.str_out, remaining = expand_slab_from_saved(
                    proc.str_out, proc.bulk, h_add, fix, saved
                )
                if remaining is not None:
                    save_atom(remaining, 'save_' + proc.name_in, proc.elmlist)
            else:
                proc.str_out = expand_slab(proc.str_out, proc.bulk, h_add, fix)
            save_atom(proc.str_out, 'add_' + proc.name_in, proc.elmlist)
            return True

        proc.logger.print("No slab modification required")
        if saved is not None and os.path.exists(saved_path):
            shutil.copy(saved_path, 'save_' + proc.name_in)
        return False

    @staticmethod
    def _chf_ratio_above(atoms, h_crit: float) -> float:
        pos_z = atoms.get_positions()[:, 2]
        syms = np.array(atoms.get_chemical_symbols())
        above = pos_z > h_crit
        if not above.any():
            return 0.0
        return float(np.isin(syms[above], list(_CHF)).sum() / above.sum())

    @staticmethod
    def _is_penetrated(atoms, h_threshold: float) -> bool:
        pos_z = atoms.get_positions()[:, 2]
        syms = atoms.get_chemical_symbols()
        return any(sym in _CHF and z < h_threshold for sym, z in zip(syms, pos_z))

    @staticmethod
    def _prev_saved_path(proc) -> str:
        interval = proc.config['run_params']['slab_update_iteration']
        return 'save_' + proc.fname_in(proc.iteration - interval)
