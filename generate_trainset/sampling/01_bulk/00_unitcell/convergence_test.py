import sys
import subprocess
import numpy as np
from ase.io import read


def read_data(path_outcars):
    energy_list = []
    force_mat_list = []
    stress_list = []

    n_atoms = len(read(path_outcars[0]))

    for path_outcar in path_outcars:
        outcar = read(path_outcar)
        energy = outcar.get_potential_energy() / n_atoms
        force = outcar.get_forces()
        energy_list.append(energy)
        force_mat_list.append(force)
        stress = subprocess.check_output(
                f"grep external {path_outcar} | awk '{{print $4}}'",
                shell=True, text=True,
                ).strip('\n')
        stress = float(stress)
        stress_list.append(stress)

    return energy_list, force_mat_list, stress_list


def check_energy_criteria(energy_list):
    energy_crit = 0.01  # 10 meV/atom
    for idx, energy_0 in enumerate(energy_list[:-2]):
        energy_1 = energy_list[idx+1]
        energy_2 = energy_list[idx+2]

        cond = (
                abs(energy_0 - energy_1) < energy_crit
                and abs(energy_1 - energy_2) < energy_crit
                and abs(energy_0 - energy_2) < energy_crit
                )

        if cond:
            return True, idx

    return False, None


def check_force_criteria(force_mat_list):
    force_crit = 0.02  # 0.02 eV/A
    for idx, force_mat_0 in enumerate(force_mat_list[:-2]):
        force_mat_1 = force_mat_list[idx+1]
        force_mat_2 = force_mat_list[idx+2]

        cond = (
                np.max(np.abs(force_mat_0 - force_mat_1)) < force_crit
                and np.max(np.abs(force_mat_1 - force_mat_2)) < force_crit
                and np.max(np.abs(force_mat_0 - force_mat_2)) < force_crit
                )

        if cond:
            return True, idx

    return False, None


def check_stress_criteria(stress_list):
    stress_crit = 10  # 10 kbar
    for idx, stress_0 in enumerate(stress_list[:-2]):
        stress_1 = stress_list[idx+1]
        stress_2 = stress_list[idx+2]

        cond = (
                abs(stress_0 - stress_1) < stress_crit
                and abs(stress_1 - stress_2) < stress_crit
                and abs(stress_0 - stress_2) < stress_crit
                )

        if cond:
            return True, idx

    return False, None


def main():
    path_outcars = [
        sys.argv[1] + f'/{i}/OUTCAR' for i in range(300, 750, 50)
    ]

    energy_list, force_mat_list, stress_list = read_data(path_outcars)
    is_converged_1, idx_1 = check_energy_criteria(energy_list)
    is_converged_2, idx_2 = check_force_criteria(force_mat_list)
    is_converged_3, idx_3 = check_stress_criteria(stress_list)

    cond = (is_converged_1 and is_converged_2 and is_converged_3)
    if cond:
        idx_converged_max = max(idx_1, idx_2, idx_3)
        result = path_outcars[idx_converged_max]

        print(result)


main()
