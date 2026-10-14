"""
Interatomic potential parameters keyed by element-symbol pairs.

To extend to a new system, add entries to D2_PARAMS and ZBL_PARAMS.
Both dicts treat (A, B) and (B, A) as equivalent; lookup functions handle both.
"""

# Grimme D2 dispersion: (C6 [J·nm^6/mol], R0 [Angstrom])
D2_PARAMS: dict[tuple[str, str], tuple[float, float]] = {
    ('Si', 'Si'): (95.674993, 3.432),
    ('Si', 'O'):  (26.347936, 3.058),
    ('Si', 'C'):  (41.659745, 3.168),
    ('Si', 'H'):  (11.783155, 2.717),
    ('Si', 'F'):  (27.272705, 3.003),
    ('O',  'O'):  (7.255958,  2.684),
    ('O',  'C'):  (11.472677, 2.794),
    ('O',  'H'):  (3.244963,  2.343),
    ('O',  'F'):  (7.510630,  2.629),
    ('C',  'C'):  (18.139896, 2.904),
    ('C',  'H'):  (5.130737,  2.453),
    ('C',  'F'):  (11.875349, 2.739),
    ('H',  'H'):  (1.451192,  2.002),
    ('H',  'F'):  (3.358856,  2.288),
    ('F',  'F'):  (7.774241,  2.574),
    # N-containing systems
    ('Si', 'N'):  (34.920462, 3.113),
    ('N',  'N'):  (12.74772,  2.794),
    ('C',  'N'):  (15.205440, 2.849),
    ('H',  'N'):  (4.30106,   2.398),
    ('F',  'N'):  (9.954622,  2.684),
}

# ZBL short-range repulsion: (inner cutoff [Å], outer cutoff [Å])
ZBL_PARAMS: dict[tuple[str, str], tuple[float, float]] = {
    ('Si', 'Si'): (1.598, 2.054),
    ('Si', 'O'):  (1.070, 1.376),
    ('Si', 'C'):  (1.208, 1.553),
    ('Si', 'H'):  (1.079, 1.388),
    ('Si', 'F'):  (1.145, 1.472),
    ('O',  'O'):  (0.863, 1.110),
    ('O',  'C'):  (0.800, 1.029),
    ('O',  'H'):  (0.691, 0.888),
    ('O',  'F'):  (0.961, 1.235),
    ('C',  'C'):  (0.920, 1.183),
    ('C',  'H'):  (0.797, 1.024),
    ('C',  'F'):  (0.908, 1.168),
    ('H',  'H'):  (0.525, 0.675),
    ('H',  'F'):  (0.657, 0.844),
    ('F',  'F'):  (0.997, 1.282),
    # N-containing systems
    ('Si', 'N'):  (1.104, 1.420),
    ('N',  'N'):  (0.780, 1.002),
    ('C',  'N'):  (1.104, 1.420),
    ('H',  'N'):  (0.735, 0.945),
    ('F',  'N'):  (0.938, 1.206),
}


def get_d2(e1: str, e2: str) -> tuple[float, float]:
    """Return D2 parameters for (e1, e2), trying both orderings."""
    if (e1, e2) in D2_PARAMS:
        return D2_PARAMS[(e1, e2)]
    if (e2, e1) in D2_PARAMS:
        return D2_PARAMS[(e2, e1)]
    raise KeyError(
        f"No D2 parameters for ({e1}, {e2}). "
        "Add an entry to params/potentials.py::D2_PARAMS."
    )


def get_zbl(e1: str, e2: str) -> tuple[float, float]:
    """Return ZBL parameters for (e1, e2), trying both orderings."""
    if (e1, e2) in ZBL_PARAMS:
        return ZBL_PARAMS[(e1, e2)]
    if (e2, e1) in ZBL_PARAMS:
        return ZBL_PARAMS[(e2, e1)]
    raise KeyError(
        f"No ZBL parameters for ({e1}, {e2}). "
        "Add an entry to params/potentials.py::ZBL_PARAMS."
    )
