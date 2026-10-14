"""
SciPy cKDTree-based connectivity graph builder.

Does not require NetworkX. Suitable for environments where NetworkX is
unavailable or when faster query times are preferred for large systems.
Only supports orthorhombic periodic cells.
"""
import numpy as np
import ase

from .base import GraphBuilderBase


class ScipyBuilder(GraphBuilderBase):
    """cKDTree nearest-neighbor search with element-pair cutoff filtering."""

    def find_edges(
        self, atoms: ase.Atoms, cutoff_matrix: np.ndarray
    ) -> list[tuple[int, int]]:
        from scipy.spatial import cKDTree

        pos = atoms.get_positions()
        cell = atoms.get_cell()
        atom_types = atoms.get_array('type') - 1
        max_cutoff = float(np.max(cutoff_matrix))

        lx, ly, lz = cell.diagonal()
        pos_wrapped = pos % np.array([lx, ly, lz])

        # cKDTree with boxsize enables periodic distance queries for orthorhombic cells.
        tree = cKDTree(pos_wrapped, boxsize=[lx, ly, lz])
        pairs = tree.query_pairs(max_cutoff, output_type='ndarray')

        if len(pairs) == 0:
            return []

        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        diff = pos_wrapped[i_idx] - pos_wrapped[j_idx]
        diff -= np.round(diff / [lx, ly, lz]) * [lx, ly, lz]
        dists = np.linalg.norm(diff, axis=1)

        cutoffs = cutoff_matrix[atom_types[i_idx], atom_types[j_idx]]
        mask = dists < cutoffs
        return list(zip(i_idx[mask].tolist(), j_idx[mask].tolist()))
