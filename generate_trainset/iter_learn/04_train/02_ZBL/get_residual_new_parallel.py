import ase
from ase.io import read, write
import subprocess as sub
import os
import torch
from braceexpand import braceexpand
import cProfile
from concurrent.futures import ThreadPoolExecutor


class CannotFindImageError(Exception):
    pass


def compress_outcar(filename):
    """
    *** From SIMPLE-NN code ***

    Compress VASP OUTCAR file for fast file-reading in ASE.
    Compressed file (tmp_comp_OUTCAR)
        is temporarily created in the current directory.

    :param str filename: filename of OUTCAR

    supported properties:

    - atom types
    - lattice vector(cell)
    - free energy
    - force
    - stress
    """
    comp_name = './tmp_comp_OUTCAR'

    with open(filename, 'r') as fil, open(comp_name, 'w') as res:
        minus_tag = 0
        line_tag = 0
        ions_key = 0
        for line in fil:
            if 'POTCAR:' in line:
                res.write(line)
            if 'POSCAR:' in line:
                res.write(line)
            elif 'ions per type' in line and ions_key == 0:
                res.write(line)
                ions_key = 1
            elif 'direct lattice vectors' in line:
                res.write(line)
                minus_tag = 3
            elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
                res.write(line)
                minus_tag = 4
            elif 'POSITION          ' in line:
                res.write(line)
                line_tag = 3
            elif 'FORCE on cell =-STRESS' in line:
                res.write(line)
                minus_tag = 15
            elif 'Iteration' in line:
                res.write(line)
            elif minus_tag > 0:
                res.write(line)
                minus_tag -= 1
            elif line_tag > 0:
                res.write(line)
                if '-------------------' in line:
                    line_tag -= 1

    return comp_name


def read_structure_list(lines):
    '''
    read all OUTCAR paths from *structures_list* file (SIMPLE-NN format)
    '''
    a = []
    for line in lines:
        if not line.strip():
            continue

        if "[" in line:
            continue

        if "{" in line:
            files, idx = line.split()
            files = [[i, idx] for i in braceexpand(files)]
            a += files
        else:
            a += [[i for i in line.split()[:2]]]

    return a


def load_poscar(path_dir):
    '''
    from the given directory, read
    1) CONTCAR
    2) POSCAR (only if when (1) fails)
    '''
    path_poscar = os.path.join(path_dir, "CONTCAR")
    try:
        outcars = [read(path_poscar)]
        return outcars
    except Exception:
        print(f"Error occurred while loading {path_poscar} ... try for POSCAR")

    path_poscar = os.path.join(path_dir, "POSCAR")
    try:
        outcars = [read(path_poscar)]
        return outcars

    except Exception:
        print(f"Error occurred while loading {path_poscar} ... Loading stop.")
        raise CannotFindImageError


def load_images(path_outcar, outcar_idx):
    '''
    for the given outcar path and index, load images
    '''
    path_dir = os.path.dirname(path_outcar)

    if outcar_idx == ':':
        try:
            return load_poscar(path_dir)
        except CannotFindImageError:
            pass

    path_outcar_compressed = compress_outcar(path_outcar)
    outcars = read(path_outcar_compressed, format='vasp-out', index=outcar_idx)
    return outcars


def generate_poscars(structure_list):
    '''
    for the given *structure_list
    '''
    path_output_dir = "structures"
    os.makedirs(path_output_dir, exist_ok=True)

    with open(structure_list, "r") as f:
        lines = f.readlines()

    path_outcar_list = read_structure_list(lines)
    n = 0

    for (path_outcar, outcar_idx) in path_outcar_list:
        # Read for oneshot
        outcars = load_images(path_outcar, outcar_idx)

        # Save each images
        for outcar in outcars:
            n += 1
            poscar_name = f"structures/POSCAR_{n}"

            if not os.path.isfile(poscar_name):
                print(f"Image generated: {poscar_name}, from {path_outcar}")
                ase.io.write(poscar_name, outcar, format='vasp')
            else:
                print(f"File already exists: {poscar_name}")


def generate_sorted_dict(sorted_names):
    sorted_dict = {}

    for e, i in enumerate(sorted_names):
        sorted_dict[i] = f"{e+1}"

    if "Si" in sorted_dict.keys():
        sorted_dict["Si"] = [sorted_dict["Si"], "14", "28.0855"]
    if "H" in sorted_dict.keys():
        sorted_dict["H"] = [sorted_dict["H"], "1", "3.0"]
    if "O" in sorted_dict.keys():
        sorted_dict["O"] = [sorted_dict["O"], "8", "16.0000"]
    if "F" in sorted_dict.keys():
        sorted_dict["F"] = [sorted_dict["F"], "9", "18.998"]
    if "C" in sorted_dict.keys():
        sorted_dict["C"] = [sorted_dict["C"], "6", "12.011"]

    return sorted_dict


def write_pair_coeff(sorted_names, sorted_dict, cutoff_dict, f):
    for e1, atom_type_1 in enumerate(sorted_names):
        for atom_type_2 in sorted_names[e1:]:
            bond_type = atom_type_1 + atom_type_2
            pair_list = [
                sorted_dict[atom_type_1][0],
                sorted_dict[atom_type_2][0],
                sorted_dict[atom_type_1][1],
                sorted_dict[atom_type_2][1],
                str(cutoff_dict[bond_type][0]),
                str(cutoff_dict[bond_type][1]),
            ]
            my_pair_coeff = " ".join(pair_list)
            f.write(f"pair_coeff  {my_pair_coeff}\n")


def write_mass(sorted_names, sorted_dict, f):
    for atom_type in sorted_names:
        atom_idx, _, mass = sorted_dict[atom_type]
        f.write(f"mass  {atom_idx} {mass}\n")


def write_movie_dump(idx, f):
    line = f"dump dMovie all custom 1 dump_{idx+1}.lammpstrj "
    line += "id type x y z fx fy fz\n"
    line += "dump_modify  dMovie  sort  id\n"
    f.write(line)


def write_lammps_input(
        lines_header, lines_footer, sorted_names, sorted_dict,
        cutoff_dict, idx):
    with open(f"lammps_{idx+1}.in", 'w') as f:
        for line in lines_header:
            if 'coo' in line:
                f.write(f"read_data    coo_{idx+1}")
                continue
            f.write(line)

        write_pair_coeff(sorted_names, sorted_dict, cutoff_dict, f)
        write_mass(sorted_names, sorted_dict, f)
        write_movie_dump(idx, f)

        for line in lines_footer:
            f.write(line)


def run_lammps_zbl(
        lmp_cmd, idx, i, lines_header, lines_footer,
        sorted_names, cutoff_dict):
    poscar = read(f"../structures/{i}", format='vasp', index=0)
    sorted_dict = generate_sorted_dict(sorted_names)

    write_lammps_input(
        lines_header, lines_footer, sorted_names, sorted_dict,
        cutoff_dict, idx)

    write(f"coo_{idx+1}", poscar, specorder=sorted_names, format='lammps-data')
    cmd_args = [
        lmp_cmd, '-sf', 'intel', '-in', f'lammps_{idx+1}.in',
        f">& lammps_{idx+1}.out"]
    sub.call(" ".join(cmd_args), shell=True)


def calculate_zbl(lmp_cmd, cutoff_dict):
    path_input_dir = "structures"
    path_output_dir = "lammps_outs"

    str_list = os.listdir(path_input_dir)
    str_list = sorted(str_list, key=lambda x: int(x.split('_')[1]))
    os.makedirs(path_output_dir, exist_ok=True)

    # Header file
    with open("lammps_front", "r") as f:
        lines_header = f.readlines()
    # Footer file
    with open("lammps_bottom", "r") as f:
        lines_footer = f.readlines()
    os.chdir("lammps_outs")

    sorted_names = ['Si', "O", 'H', 'F', 'C']

    # Using ThreadPoolExecutor to parallelize the loop
    with ThreadPoolExecutor() as executor:
        executor.map(
            run_lammps_zbl, [lmp_cmd]*len(str_list),
            range(len(str_list)), str_list,
            [lines_header]*len(str_list), [lines_footer]*len(str_list),
            [sorted_names]*len(str_list), [cutoff_dict]*len(str_list))

    os.chdir("../")


def generate_new_pt(pt_path, zbl_path, to_save_path):
    path_output_dir = "data_residual"
    os.makedirs(path_output_dir, exist_ok=True)

    pt_list = os.listdir(pt_path)

    for i in pt_list:
        e = int(i.split(".")[0][4:])
        my_path = zbl_path + f"/lammps_{e}.out -A1 |tail -n1"
        cmd = " ".join(["grep", "TotEng", my_path])
        zbl_dat = sub.check_output(
                cmd, shell=True, universal_newlines=True).split()

        dat = float(zbl_dat[1])
        E_zbl = torch.tensor(dat, dtype=torch.float64)

        dat = [float(pressure) for pressure in zbl_dat[2:]]
        S_zbl = torch.tensor(dat, dtype=torch.float64) / 1000

        my_path = zbl_path+f"/dump_{e}.lammpstrj"
        zbl_trj = read(my_path, format='lammps-dump-text', index=0)
        dat = zbl_trj.get_forces()
        F_zbl = torch.tensor(dat, dtype=torch.float64)

        DFT_pt = torch.load(pt_path+'/'+i)
        DFT_pt['E'] -= E_zbl
        DFT_pt['S'] -= S_zbl
        DFT_pt['F'] -= F_zbl

        torch.save(DFT_pt, to_save_path+'/'+i)


def main():
    pt_path = "../01_gen_pt/data"  # folder that contains *.pt files
    zbl_path = './lammps_outs'
    to_save_path = './data_residual'
    structure_list = "structure_list"  # SIMPLE-NN structure_list
    lmp_cmd = "/home/andynn/lammps/build_etch_d2/lmp"

    # VERSION OF ZBL/PAIR

    # CUTOFF distance that use
    cutoff_dict = {
        'CC': [0.920, 1.183],
        'CF': [0.908, 1.168],
        'FC': [0.908, 1.168],
        'CH': [0.797, 1.024],
        'HC': [0.797, 1.024],
        'CSi': [1.208, 1.553],
        'SiC': [1.208, 1.553],
        'FF': [0.997, 1.282],
        'HF': [0.657, 0.844],
        'FH': [0.657, 0.844],
        'HH': [0.525, 0.675],
        'OC': [0.800, 1.029],
        'CO': [0.800, 1.029],
        'OF': [0.961, 1.235],
        'FO': [0.961, 1.235],
        'OH': [0.691, 0.888],
        'HO': [0.691, 0.888],
        'OO': [0.863, 1.110],
        'OSi': [1.070, 1.376],
        'SiO': [1.070, 1.376],
        'SiF': [1.145, 1.472],
        'FSi': [1.145, 1.472],
        'SiH': [1.079, 1.388],
        'HSi': [1.079, 1.388],
        'SiSi': [1.598, 2.054],
    }

    poscar_exist = False
    calced = False
    ext_pt = True

    if not poscar_exist:
        generate_poscars(structure_list)

    if not calced:
        calculate_zbl(lmp_cmd, cutoff_dict)

    if ext_pt:
        generate_new_pt(pt_path, zbl_path, to_save_path)


cProfile.run("main()", "result.prof")
