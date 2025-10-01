import numpy as np
import os
import sys
from ase.io import read, write
from concurrent.futures import ThreadPoolExecutor


def get_z_max(snap):
    '''
    Get the maximum z-coordinate of the atoms in the snapshot
    '''
    maxpos = snap.get_positions()
    z_max = np.max(maxpos[:, 2])

    return z_max


def get_snap_list(src, dst, i, **params):
    '''
    Get the list of snapshots to be processed
    '''
    coll_st = params["coll_st"]
    coll_end = params["coll_end"]
    coll_step = params["coll_step"]
    rlx_step = params["rlx_step"]

    os.makedirs(f"structures/{i}", exist_ok=True)
    file_name = f"{src}/dump_{i}.lammps"
    atom = read(file_name, format='lammps-dump-text', index="1:")

    # COLL
    snap_list = [i for i in range(coll_st, coll_end, coll_step)] +\
                [i for i in range(coll_end, len(atom), rlx_step)]
    snap_list = np.array(snap_list)

    return atom, snap_list


def process_snapshot(dst, snapshot, i, dir_idx, snap_idx, z_cutoff):
    '''
    Process the snapshot and write the POSCAR
    '''
    # To Check not reactive cases
    z_max = get_z_max(snapshot)
    if z_max > z_cutoff:
        z_max = z_cutoff

    poscar = snapshot.copy()
    while poscar:
        poscar.pop()
    for atom in snapshot:
        if atom.position[2] > z_max or atom.position[2] < 0.0:
            continue
        poscar.append(atom)
    snapshot = poscar

    cell = snapshot.get_cell()
    cell[2][2] = z_max + 10.0
    snapshot.set_cell(cell)

    os.makedirs(f"{dst}/{i}", exist_ok=True)
    write(
        f"{dst}/{i}/POSCAR_{dir_idx}", snapshot,
        format='vasp', sort=True,
        label=f"incidence:{i} / step {snap_idx+1}",
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python read_write_mod.py <path_to_dump_folder>")
        sys.exit(1)

    src = sys.argv[1]
    params = {
        "total_iter": 25,

        "coll_st": 0,
        "coll_end": 120,
        "coll_step": 20,

        "rlx_step": 80,
        "z_cutoff": 30.0,

        "n_images_max": 10,
    }
    dst = "structures"

    for idx_inc in range(1, params["total_iter"]+1):
        print(f"currently incidence {idx_inc}")
        dir_idx = 0
        atom, snap_list = get_snap_list(src, dst, idx_inc, **params)

        # truncate the list if it is too long
        snap_list = snap_list[:params["n_images_max"]]

        n_images = len(snap_list)

        with ThreadPoolExecutor() as executor:
            executor.map(
                process_snapshot,
                [dst]*n_images,
                [atom[i] for i in snap_list],
                [idx_inc]*n_images,
                range(dir_idx, dir_idx+n_images+1),
                snap_list,
                [params["z_cutoff"]]*n_images,
                )

        dir_idx += n_images

        print(f"{dst}/{idx_inc}/ written with {n_images} POSCARs")


if __name__ == "__main__":
    main()
