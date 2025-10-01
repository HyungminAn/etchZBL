from ase.io import read, write
import subprocess as sub
import os
import cProfile
from concurrent.futures import ThreadPoolExecutor


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
                "zbl/pair",
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


def run_lammps(
        lmp_cmd, idx, path_poscar, lines_header, lines_footer,
        sorted_names, cutoff_dict):
    poscar = read(path_poscar)
    sorted_dict = generate_sorted_dict(sorted_names)

    write_lammps_input(
        lines_header, lines_footer, sorted_names, sorted_dict,
        cutoff_dict, idx)

    write(f"coo_{idx+1}", poscar, specorder=sorted_names, format='lammps-data')
    cmd_args = [
        lmp_cmd, '-sf', 'intel', '-in', f'lammps_{idx+1}.in',
        f">& lammps_{idx+1}.out"]
    sub.call(" ".join(cmd_args), shell=True)


def get_poscar_paths(input_path):
    poscar_list = []
    ion_type_list = [
        'CF_10',
        'CF_30',
        'CF3_10',
        'CF3_30',
        'CH2F_10',
        'CH2F_30',
        'CHF2_10',
        'CHF2_30',
    ]
    for ion_type in ion_type_list:
        for i in range(1, 51):
            pos_list_within_dir = []
            for dirpath, dirnames, filenames in os.walk(
                    f"{input_path}/{ion_type}/{i}"):
                for filename in filenames:
                    if filename.startswith('POSCAR_'):
                        full_path = os.path.abspath(
                            os.path.join(dirpath, filename))
                        pos_list_within_dir.append(full_path)

            pos_list_within_dir = sorted(
                    pos_list_within_dir,
                    key=lambda x: int(x.split('/')[-1].split('_')[1])
                    )
            poscar_list += pos_list_within_dir
    return poscar_list


def run_lammps_batch(input_path, output_path, lmp_cmd, cutoff_dict):
    path_poscars = get_poscar_paths(input_path)
    os.makedirs(output_path, exist_ok=True)

    # Header file
    with open("lammps_front", "r") as f:
        lines_header = f.readlines()
    # Footer file
    with open("lammps_bottom", "r") as f:
        lines_footer = f.readlines()
    os.chdir(output_path)

    sorted_names = ["Si", "O", "C", "H", "F"]

    # Using ThreadPoolExecutor to parallelize the loop
    n = len(path_poscars)
    with ThreadPoolExecutor() as executor:
        executor.map(
            run_lammps, [lmp_cmd] * n, range(n), path_poscars,
            [lines_header] * n, [lines_footer] * n,
            [sorted_names] * n, [cutoff_dict] * n)

    os.chdir("../")


def main():
    input_path = '../structures'
    output_path = './lammps_outs'
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

    run_lammps_batch(input_path, output_path, lmp_cmd, cutoff_dict)


cProfile.run("main()", "result.prof")
