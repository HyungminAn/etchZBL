import os
import shutil


def main():
    src = '../00_gen_molecules'
    folder_list = [
        i for i in os.listdir(src)
        if os.path.isdir(f'{src}/{i}')]

    unstable_list = [
        'CFO', 'CFO3', 'CH2O2', 'CHFO2', 'CHO3', 'FO2', 'H2O2Si', 'HO3Si',
    ]

    folder_list = [
        i for i in folder_list
        if i not in unstable_list
    ]

    for i in folder_list:
        os.makedirs(i, exist_ok=True)
        shutil.copy(f'{src}/{i}/CONTCAR', f'{i}/POSCAR')
        shutil.copy(f'{src}/{i}/POTCAR', f'{i}/POTCAR')


main()
