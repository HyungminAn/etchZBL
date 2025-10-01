import numpy as np
import os
from ase.io import read, write
from concurrent.futures import ThreadPoolExecutor


def get_z_max(snap):
    maxpos = snap.get_positions()
    z_max = np.max(maxpos[:, 2])

    return z_max


def get_snap_list(i, coll_st, coll_end, coll_step, rlx_step):
    os.makedirs(f"structures/{i}", exist_ok=True)
    file_name = f"./dump_{i}.lammps"
    atom = read(file_name, format='lammps-dump-text', index="1:")

    # COLL
    snap_list = list(range(coll_st,  coll_end,  coll_step))
    snap_list += list(range(coll_end, len(atom), rlx_step))
    snap_list = np.array(snap_list)

    return atom, snap_list


def process_snapshot(snapshot, i, dir_idx, snap_idx, z_cutoff):
    # To Check not reactive cases
    z_max = get_z_max(snapshot)
    cond = z_max > z_cutoff
    if cond:
        # return
        print(f"z_cutoff exceeded in {i}/POSCAR_{dir_idx}")

    cell = snapshot.get_cell()
    cell[2][2] = z_max + 10.0
    snapshot.set_cell(cell)

    os.makedirs(f"structures/{i}/poscars", exist_ok=True)
    write(
        f"structures/{i}/poscars/POSCAR_{dir_idx}", snapshot,
        format='vasp', sort=True,
        label=f"incidence:{i} / step {snap_idx+1}"
    )


def main():
    total_iter = 50

    coll_st = 0
    coll_end = 120
    coll_step = 40
    rlx_step = 80

    z_cutoff = 30.0

    for idx_inc in range(1, total_iter+1):
        print(f"currently incidence {idx_inc}")
        dir_idx = 0
        atom, snap_list = get_snap_list(
            idx_inc, coll_st, coll_end, coll_step, rlx_step)
        # Use 5 snapshots only
        snap_list = snap_list[:5]

        with ThreadPoolExecutor() as executor:
            executor.map(
                process_snapshot, [atom[i] for i in snap_list],
                [idx_inc]*len(snap_list),
                range(dir_idx, dir_idx+len(snap_list)+1),
                snap_list, [z_cutoff]*len(snap_list)
                )

        dir_idx += len(snap_list)

        print(f"structures/{idx_inc}/ written with {len(snap_list)} POSCARs")


main()
