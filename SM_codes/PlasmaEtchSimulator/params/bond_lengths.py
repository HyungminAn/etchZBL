"""
Bond cutoff distances for connectivity graph construction.

Keyed by unordered element-symbol pairs; get_cutoff() handles both orderings.
To add a new system, insert additional entries into BOND_CUTOFFS.
"""
import numpy as np

# Upper-bound bond cutoff distances in Angstrom.
BOND_CUTOFFS: dict[tuple[str, str], float] = {
    ('Si', 'Si'): 2.96712,
    ('Si', 'O'):  1.98784,
    ('Si', 'C'):  2.24319,
    ('Si', 'H'):  2.00469,
    ('Si', 'F'):  2.12597,
    ('O',  'O'):  1.60294,
    ('O',  'C'):  1.48655,
    ('O',  'H'):  1.28316,
    ('O',  'F'):  1.78380,
    ('C',  'C'):  1.70875,
    ('C',  'H'):  1.47971,
    ('C',  'F'):  1.68691,
    ('H',  'H'):  0.97536,
    ('H',  'F'):  1.21950,
    ('F',  'F'):  1.85160,
    # N-containing systems (Si3N4 etc.)
    ('Si', 'N'):  2.24319,
    ('O',  'N'):  1.50000,
    ('C',  'N'):  1.47971,
    ('H',  'N'):  1.08000,
    ('F',  'N'):  1.48000,
    ('N',  'N'):  1.45000,
}


def get_cutoff(e1: str, e2: str) -> float:
    """Return the bond cutoff distance for element pair (e1, e2)."""
    if (e1, e2) in BOND_CUTOFFS:
        return BOND_CUTOFFS[(e1, e2)]
    if (e2, e1) in BOND_CUTOFFS:
        return BOND_CUTOFFS[(e2, e1)]
    raise KeyError(
        f"No bond cutoff defined for ({e1}, {e2}). "
        "Add an entry to params/bond_lengths.py::BOND_CUTOFFS."
    )


def build_cutoff_matrix(elmlist: list[str]) -> np.ndarray:
    """Build an (N x N) float64 cutoff matrix indexed by element position in elmlist."""
    n = len(elmlist)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, e1 in enumerate(elmlist):
        for j, e2 in enumerate(elmlist):
            matrix[i, j] = get_cutoff(e1, e2)
    return matrix
