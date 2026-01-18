import os
import subprocess
import shutil


def find_directories_with_poscars(root_directory):
    '''
    Define a function to find directories with 'poscars' folder
    '''
    poscars_directories = []

    for root, dirs, _ in os.walk(root_directory):
        if 'poscars' in dirs and 'inputs' not in root:
            poscars_directories.append(os.path.abspath(root))

    return poscars_directories


def check_converged(path_OUTCAR):
    command_converged = \
        f"grep 'aborting loop because EDIFF is reached' {path_OUTCAR} | wc -l"
    output = subprocess.check_output(command_converged, shell=True, text=True)
    result = int(output.strip())
    if result:
        is_converged = True
    else:
        is_converged = False

    return is_converged


def check_max_iter(path_OUTCAR):
    max_iter = 300

    command_max_iter = f"grep 'Iteration ' {path_OUTCAR} | wc -l"
    output = subprocess.check_output(command_max_iter, shell=True, text=True)
    result = int(output.strip())
    if result == max_iter:
        is_max_iter = True
    else:
        is_max_iter = False

    return is_max_iter


def check_all_poscars_done(directory):
    poscars_dir = os.path.join(directory, 'poscars')
    n_poscars_to_calculate = len(
        [i for i in os.listdir(poscars_dir) if 'POSCAR_' in i])
    n_poscars_calculated = len(
        [i for i in os.listdir(directory) if 'POSCAR_' in i])

    cond = (n_poscars_to_calculate > n_poscars_calculated)
    if cond:
        print(f" ****** {directory} not all done")

    return cond


def find_undone_outcars(directory):
    outcar_list = [
        f'{directory}/{i}/OUTCAR' for i in os.listdir(directory)
        if 'POSCAR_' in i]
    undone_outcar_list = []

    for outcar in outcar_list:
        is_converged = check_converged(outcar)
        is_max_iter = check_max_iter(outcar)

        if is_converged:
            continue

        if is_max_iter:
            continue
        else:
            undone_outcar_list.append(outcar)

    return undone_outcar_list


def find_unconverged_outcars(directory):
    outcar_list = [
        f'{directory}/{i}/OUTCAR' for i in os.listdir(directory)
        if 'POSCAR_' in i]
    unconverged_outcar_list = []

    for outcar in outcar_list:
        is_converged = check_converged(outcar)
        is_max_iter = check_max_iter(outcar)

        if is_converged:
            continue

        if is_max_iter:
            unconverged_outcar_list.append(outcar)

    return unconverged_outcar_list


def gather_undone_poscars(directory):
    # find where does calculation stopped
    undone_outcar_list = find_undone_outcars(directory)
    poscar_list = [i.split('/')[-2] for i in undone_outcar_list]
    poscars_before_calc = [
        i for i in os.listdir(f"{directory}/poscars")
        if i not in os.listdir(directory)
    ]
    poscars_all = poscar_list + poscars_before_calc

    # Undone POSCARs
    dst = f"{directory}/poscars_continue"
    os.makedirs(dst, exist_ok=True)

    for poscar in poscars_all:
        shutil.copy(f"{directory}/poscars/{poscar}", dst)

    shutil.copy(f"{directory}/INCAR", dst)
    shutil.copy(f"{directory}/KPOINTS", dst)
    shutil.copy(f"{directory}/POTCAR", dst)


def gather_unconverged_poscars(directory):
    # Unconverged POSCARs
    unconverged_outcar_list = find_unconverged_outcars(directory)
    poscars_unconverged_list = [
        i.split('/')[-2] for i in unconverged_outcar_list
    ]
    dst = f"{directory}/poscars_unconverged"
    # Skip if all poscars are converged
    if not unconverged_outcar_list:
        return

    os.makedirs(dst, exist_ok=True)

    for poscar in poscars_unconverged_list:
        shutil.copy(f"{directory}/poscars/{poscar}", dst)

    shutil.copy(f"{directory}/INCAR", dst)
    shutil.copy(f"{directory}/KPOINTS", dst)
    shutil.copy(f"{directory}/POTCAR", dst)


def main():
    # Specify the root directory where you want to start the search
    root_directory = os.getcwd()

    # Call the function to find directories with 'poscars' folder
    result = find_directories_with_poscars(root_directory)

    # Print the result
    for directory in result:
        cond = check_all_poscars_done(directory)

        # There exists undone poscars
        if cond:
            gather_undone_poscars(directory)

        gather_unconverged_poscars(directory)


main()
