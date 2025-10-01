import os
from ase.io import read, write
# from ase.build import sort
# from ase.constraints import FixAtoms


def main():
    # matcher = {1:14,2:7,3:1,4:9}
    src = "post_process_bulk_gas"
    read_options = {
        "format": "lammps-data",
        "style": "atomic",
        "index": "0",
    }
    fix_h = 2.0

    folders = [os.path.join(src, i) for i in os.listdir(src) if "gas" not in i]
    for i in folders:
        for j in os.listdir(i):
            image = read(f'{i}/{j}/coo', **read_options)
            # image = sort(image,tags=image.arrays['id'])
            # image.set_atomic_numbers([matcher[i] for i in image.get_atomic_numbers()])
            # arg_sort=image.numbers.argsort()
    #
            # selective_idx=[]
            # for atom_idx in range(len(image)):
            #     if image.positions[atom_idx,2]<2:
            #         selective_idx.append(atom_idx)
            # s_flag = FixAtoms(selective_idx)
            s_flag = [atom.index for atom in image if atom.position[2] < fix_h]
            image.set_constraint(s_flag)
            # write(f'{i}/{j}/POSCAR',image[arg_sort],format='vasp')
            write(f'{i}/{j}/POSCAR',image,format='vasp', sort=True)


if __name__ == '__main__':
    main()
