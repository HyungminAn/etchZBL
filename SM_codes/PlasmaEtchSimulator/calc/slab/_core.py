"""
Low-level slab expansion and trimming helpers.
No torch dependency; uses numpy only.
"""
import copy
import os

import numpy as np
import ase
import ase.build

from ..io import save_atom, load_atom


def wrap_positions(atoms: ase.Atoms) -> np.ndarray:
    """Return positions wrapped into the primary orthorhombic PBC cell."""
    pos = atoms.get_positions().copy()
    for i in range(3):
        pos[:, i] = pos[:, i] % atoms.cell[i, i]
    return pos


def get_slab_height(clusters: list[set], atoms: ase.Atoms) -> float:
    """Return the 95th-percentile Z of the largest connected cluster."""
    largest = max(clusters, key=len)
    pos_z = atoms.get_positions()[list(largest), 2]
    return float(np.percentile(pos_z, 95))


MIN_MATCH_ATOMS = 6
MATCH_TOL = 5e-4


def find_bulk_origin(pos_slab: np.ndarray,
                     pos_bulk: np.ndarray,
                     fix_height: float) -> tuple[float, int]:
    """
    Match atoms in the slab fixed layer against bulk XY positions to find the
    Z coordinate where bulk atoms should be anchored.
    """
    in_fix = np.where(pos_slab[:, 2] < fix_height)[0]
    sorted_idx = in_fix[np.argsort(pos_slab[in_fix, 2])][:MIN_MATCH_ATOMS]

    for slab_idx in sorted_idx:
        xy = pos_slab[slab_idx, :2]
        diff_xy = np.abs(pos_bulk[:, :2] - xy)
        match = np.all(diff_xy < MATCH_TOL, axis=1)
        if match.any():
            bulk_idx = int(np.argmax(match))
            return float(pos_slab[slab_idx, 2]), bulk_idx

    raise RuntimeError(
        "Cannot locate a matching bulk origin atom within the slab fixed layer. "
        "Verify that bulk_loc and slab_loc correspond to the same crystal structure."
    )


def expand_slab(slab: ase.Atoms,
                bulk: ase.Atoms,
                h_add: float,
                fix_height: float) -> ase.Atoms:
    """Return a new slab extended downward by h_add Angstrom of bulk material."""
    slab = copy.deepcopy(slab)
    pos_slab = wrap_positions(slab)
    pos_bulk = wrap_positions(bulk)

    origin_z, bulk_idx = find_bulk_origin(pos_slab, pos_bulk, fix_height)
    bulk_h_origin = float(pos_bulk[bulk_idx, 2])
    bulk_z_cell = float(bulk.cell[2, 2])

    slab.positions[:, 2] += h_add
    shift = origin_z + h_add - bulk_h_origin

    lo = bulk_h_origin - h_add - MATCH_TOL
    hi = bulk_h_origin - MATCH_TOL
    for idx in np.where((pos_bulk[:, 2] > lo) & (pos_bulk[:, 2] < hi))[0]:
        atom = copy.deepcopy(bulk[int(idx)])
        atom.position[2] += shift
        slab.append(atom)

    remaining = h_add - bulk_h_origin
    current_top = origin_z + remaining
    while remaining > 0:
        if remaining >= bulk_z_cell:
            for idx in range(len(bulk)):
                atom = copy.deepcopy(bulk[idx])
                atom.position[2] += current_top - bulk_z_cell
                slab.append(atom)
            current_top -= bulk_z_cell
            remaining -= bulk_z_cell
        else:
            for idx in np.where(pos_bulk[:, 2] > bulk_z_cell - remaining)[0]:
                atom = copy.deepcopy(bulk[int(idx)])
                atom.position[2] += current_top - bulk_z_cell
                slab.append(atom)
            break

    return ase.build.sort(slab)


def expand_slab_from_saved(slab: ase.Atoms,
                            bulk: ase.Atoms,
                            h_add: float,
                            fix_height: float,
                            saved: ase.Atoms) -> tuple[ase.Atoms, 'ase.Atoms | None']:
    """Expand slab using saved atoms first, supplementing with bulk if needed."""
    h_saved = float(saved.cell[2, 2])

    if h_saved >= h_add:
        h_remain = h_saved - h_add
        keep_mask = saved.positions[:, 2] > h_remain
        slice_atoms = saved[keep_mask]

        slab = copy.deepcopy(slab)
        slab.positions[:, 2] += h_add
        for atom in slice_atoms:
            a = copy.deepcopy(atom)
            a.position[2] -= h_remain
            slab.append(a)
        slab = ase.build.sort(slab)

        remaining = copy.deepcopy(saved[~keep_mask])
        remaining.cell[2, 2] = h_remain
        return slab, remaining

    else:
        slab = copy.deepcopy(slab)
        slab.positions[:, 2] += h_saved
        for atom in saved:
            slab.append(copy.deepcopy(atom))
        h_still_needed = h_add - h_saved
        slab = expand_slab(slab, bulk, h_still_needed, fix_height)
        return slab, None


def trim_slab(slab: ase.Atoms,
              h_sub: float,
              saved: 'ase.Atoms | None' = None) -> tuple[ase.Atoms, ase.Atoms]:
    """Remove the bottom h_sub Angstrom of the slab."""
    slab = copy.deepcopy(slab)
    keep_mask = slab.positions[:, 2] > h_sub
    removed = slab[~keep_mask]

    if saved is not None:
        archive = copy.deepcopy(saved)
        archive_h = float(archive.cell[2, 2])
        archive.cell[2, 2] += h_sub
        for atom in removed:
            a = copy.deepcopy(atom)
            a.position[2] += archive_h
            archive.append(a)
    else:
        archive = ase.Atoms(cell=slab.cell.copy())
        archive.cell[2, 2] = h_sub
        for atom in removed:
            archive.append(copy.deepcopy(atom))

    trimmed = slab[keep_mask]
    trimmed.positions[:, 2] -= h_sub
    trimmed = ase.build.sort(trimmed)
    return trimmed, archive
