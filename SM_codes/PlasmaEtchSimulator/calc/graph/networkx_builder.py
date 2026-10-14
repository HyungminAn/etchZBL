"""
NetworkX-based connectivity graph builder.

Uses spatial binning to reduce O(N^2) distance comparisons to near-linear,
then builds connected components with NetworkX.
"""
import numpy as np
import networkx as nx
import ase
from ase.geometry import get_distances

from .base import GraphBuilderBase


class NetworkXBuilder(GraphBuilderBase):
    """Spatial-binning edge finder + NetworkX connected components."""

    def find_edges(
        self, atoms: ase.Atoms, cutoff_matrix: np.ndarray
    ) -> list[tuple[int, int]]:
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        atom_types = atoms.get_array('type') - 1  # 0-indexed

        bins, bin_dict = self._split_into_bins(pos, cell, cutoff_matrix)
        x_bins, y_bins, z_bins = bins
        edges: list[tuple[int, int]] = []

        for current_bin, cur_indices in bin_dict.items():
            neighbors = [
                (
                    ((current_bin[0] + dx - 1) % len(x_bins)) + 1,
                    ((current_bin[1] + dy - 1) % len(y_bins)) + 1,
                    ((current_bin[2] + dz - 1) % len(z_bins)) + 1,
                )
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]
            ]
            neighbors = [b for b in neighbors if b in bin_dict]
            neigh_indices = [idx for b in neighbors for idx in bin_dict[b]]
            if not neigh_indices:
                continue

            _, D = get_distances(pos[cur_indices], pos[neigh_indices], cell=cell, pbc=True)
            cutoffs = cutoff_matrix[atom_types[cur_indices]][:, atom_types[neigh_indices]]
            row, col = np.where(D < cutoffs)
            for r, c in zip(row, col):
                edges.append((cur_indices[r], neigh_indices[c]))

        return edges

    def build(self, atoms: ase.Atoms, cutoff_matrix: np.ndarray) -> list[set]:
        """Build clusters via NetworkX connected_components."""
        edges = self.find_edges(atoms, cutoff_matrix)
        G = nx.Graph()
        G.add_nodes_from(range(len(atoms)))
        G.add_edges_from(edges)
        return list(nx.connected_components(G))

    @staticmethod
    def _split_into_bins(
        pos: np.ndarray, cell: np.ndarray, cutoff_matrix: np.ndarray
    ) -> tuple:
        max_cutoff = float(np.max(cutoff_matrix))
        cell_x, cell_y = cell.diagonal()[:2]

        x_bins = np.arange(0, cell_x, max_cutoff)[:-1]
        y_bins = np.arange(0, cell_y, max_cutoff)[:-1]
        z_bins = np.arange(0, float(np.max(pos[:, 2])) + max_cutoff, max_cutoff)

        xi = np.digitize(pos[:, 0], x_bins)
        yi = np.digitize(pos[:, 1], y_bins)
        zi = np.digitize(pos[:, 2], z_bins)

        bin_dict: dict = {}
        for i, key in enumerate(zip(xi, yi, zi)):
            bin_dict.setdefault(key, []).append(i)
        return (x_bins, y_bins, z_bins), bin_dict
