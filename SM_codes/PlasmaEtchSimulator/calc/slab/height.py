"""
Height-based slab modification strategy.

Maintains total slab height within [slab_min_height, slab_max_height].
"""
import os
import shutil

from ..io import save_atom, load_atom
from ._core import get_slab_height, expand_slab, expand_slab_from_saved, trim_slab


class HeightBasedModifier:
    """
    Add or remove slab material based on the absolute slab height.
    - height < slab_min_height  → expand toward slab_center_height
    - height > slab_max_height  → trim toward slab_center_height
    """

    def run(self, proc) -> bool:
        """Modify proc.str_out to keep slab height in range.
        Returns True when annealing of the modified slab is required."""
        if proc.calc_done():
            return False

        rp = proc.config['run_params']
        ep = proc.config['etch_params']
        h_min = rp['slab_min_height']
        h_max = rp['slab_max_height']
        h_center = rp['slab_center_height']
        fix = ep['fix']

        h = get_slab_height(proc.clusters_in, proc.str_in)
        saved_path = self._prev_saved_path(proc)
        saved = load_atom(saved_path, proc.convert_dict) if os.path.exists(saved_path) else None

        if h < h_min:
            h_add = h_center - h
            proc.logger.print(f"Slab too short ({h:.2f} Å < {h_min} Å); expanding by {h_add:.2f} Å")
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

        if h > h_max:
            h_sub = h - h_center
            proc.logger.print(f"Slab too tall ({h:.2f} Å > {h_max} Å); trimming by {h_sub:.2f} Å")
            proc.str_out, archive = trim_slab(proc.str_out, h_sub, saved)
            save_atom(proc.str_out, 'sub_' + proc.name_in, proc.elmlist)
            save_atom(archive, 'save_' + proc.name_in, proc.elmlist)
            return False

        proc.logger.print(f"Slab height OK ({h_min} Å < {h:.2f} Å < {h_max} Å)")
        if saved is not None:
            shutil.copy(saved_path, 'save_' + proc.name_in)
        return False

    @staticmethod
    def _prev_saved_path(proc) -> str:
        interval = proc.config['run_params']['slab_update_iteration']
        return 'save_' + proc.fname_in(proc.iteration - interval)
