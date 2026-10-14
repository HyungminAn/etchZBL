"""
Abstract base class for connectivity graph builders.
"""
from abc import ABC, abstractmethod

import numpy as np
import ase


class GraphBuilderBase(ABC):
    """Build a bond connectivity graph for a set of atoms."""

    @abstractmethod
    def find_edges(
        self, atoms: ase.Atoms, cutoff_matrix: np.ndarray
    ) -> list[tuple[int, int]]:
        """Return bonded (i, j) index pairs within element-pair cutoff distances."""

    def build_clusters(
        self, edges: list[tuple[int, int]], n_atoms: int
    ) -> list[set]:
        """Group atoms into connected components using path-compressed union-find."""
        parent = list(range(n_atoms))
        rank = [0] * n_atoms

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        for i, j in edges:
            union(i, j)

        cluster_map: dict[int, set] = {}
        for idx in range(n_atoms):
            cluster_map.setdefault(find(idx), set()).add(idx)
        return list(cluster_map.values())

    def build(self, atoms: ase.Atoms, cutoff_matrix: np.ndarray) -> list[set]:
        """Return a list of atom-index sets, one per connected component."""
        edges = self.find_edges(atoms, cutoff_matrix)
        return self.build_clusters(edges, len(atoms))
