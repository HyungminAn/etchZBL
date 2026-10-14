"""
Byproduct detection and removal after each MD step.
"""
import copy
import re

import numpy as np
import ase
from ase.data import chemical_symbols as ase_chemical_symbols

from .io import save_atom


class ByProductRemover:
    """
    Identify and remove reaction byproducts from the post-MD structure.

    A cluster is removed if:
      - Its chemical formula matches a species in byproduct_list, OR
      - Its minimum Z position is more than remove_high_cluster_height_crit
        Angstrom above the top of the main slab cluster.
    Clusters whose minimum Z is below fix_height are never removed.
    """

    def __init__(self, elmlist: list, byproduct_list: list, logger=None) -> None:
        self.elmlist = elmlist
        self.byproduct_list = byproduct_list
        self._byproduct_labels: dict[str, str] = {}
        self._parse_byproducts()
        self.logger = logger

    def run(self, proc) -> None:
        """Remove byproducts from proc.str_out in place."""
        if proc.calc_done():
            return

        remove_height_crit = proc.etch_params.get('remove_high_cluster_height_crit', 10)
        fix_height = proc.etch_params['fix']

        proc.str_out = self._remove_byproduct(
            proc.str_out,
            proc.clusters_in,
            remove_height_crit,
            fix_height,
        )
        save_atom(proc.str_out, 'rm_byproduct_' + proc.name_in, proc.elmlist)
        proc.byproduct_removed = True

    def _parse_byproducts(self) -> None:
        pattern = re.compile(
            r'(?P<element>' + '|'.join(re.escape(e) for e in self.elmlist) + r')(?P<count>\d*)'
        )
        for molecule in self.byproduct_list:
            matches = pattern.findall(molecule)
            ndict = {elm: int(cnt) if cnt else 1 for elm, cnt in matches}
            label = self._ndict_to_label(ndict)
            self._byproduct_labels[label] = molecule

    def _remove_byproduct(
        self,
        atoms: ase.Atoms,
        clusters: list[set],
        height_crit: float,
        fix_height: float,
    ) -> ase.Atoms:
        out = copy.deepcopy(atoms)
        pos_z = atoms.get_positions()[:, 2]
        slab_idx = int(np.argmax([len(c) for c in clusters]))
        h_slab_top = float(np.max(pos_z[list(clusters[slab_idx])]))

        to_remove: list[int] = []
        for idx, cluster in enumerate(clusters):
            if idx == slab_idx:
                continue
            h_min = float(np.min(pos_z[list(cluster)]))
            if h_min < fix_height:
                continue
            label, is_byproduct = self._classify_cluster(cluster, atoms)
            if self.logger:
                self.logger.print(f'  Cluster {idx}: {label} | byproduct={is_byproduct}')
            if is_byproduct or h_min - h_slab_top > height_crit:
                to_remove.extend(cluster)

        del out[to_remove]
        return out

    def _classify_cluster(self, cluster: set, atoms: ase.Atoms) -> tuple[str, bool]:
        ndict: dict[str, int] = {}
        for i in cluster:
            sym = atoms[i].symbol
            ndict[sym] = ndict.get(sym, 0) + 1
        label = self._ndict_to_label(ndict)
        return label, label in self._byproduct_labels

    @staticmethod
    def _ndict_to_label(ndict: dict) -> str:
        return ''.join(f'{sym}{cnt}' for sym, cnt in sorted(ndict.items()))
