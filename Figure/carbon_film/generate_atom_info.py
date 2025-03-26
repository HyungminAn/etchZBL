import os
import pickle
import sys

from ase.io import read

from AtomInfo import AtomInfo, AtomDict, nested_dict
from utils import timeit
from utils import read_structure as read


@timeit
def load_coo(src, struct_idx):
    path_coo = os.path.join(src, f'str_shoot_{struct_idx}.coo')
    path_coo_rm = os.path.join(src, f'rm_byproduct_str_shoot_{struct_idx}.coo')
    path_coo_add = os.path.join(src, f'add_str_shoot_{struct_idx}.coo')
    path_coo_sub = os.path.join(src, f'sub_str_shoot_{struct_idx}.coo')
    path_coo_save = os.path.join(src, f'save_str_shoot_{struct_idx}.coo')
    path_coo_before_anneal = os.path.join(src, f'str_shoot_{struct_idx}_after_mod_before_anneal.coo')
    path_coo_final = os.path.join(src, f'str_shoot_{struct_idx}_after_mod.coo')

    if not os.path.exists(path_coo):
        raise FileNotFoundError(f'{path_coo} does not exist')

    coo = read(path_coo)
    coo.wrap()
    coo_rm = read(path_coo_rm)
    coo_rm.wrap()

    if os.path.exists(path_coo_add):
        coo_add = read(path_coo_add)
        coo_add.wrap()
        coo_before_anneal = read(path_coo_before_anneal)
        coo_before_anneal.wrap()
    else:
        coo_add = None
        coo_before_anneal = None

    if os.path.exists(path_coo_sub):
        coo_sub = read(path_coo_sub)
        coo_sub.wrap()
        coo_save = read(path_coo_save)
        coo_save.wrap()
    else:
        coo_sub = None
        coo_save = None

    coo_final = read(path_coo_final)
    coo_final.wrap()

    result = {
        'coo': coo,
        'coo_rm': coo_rm,
        'coo_add': coo_add,
        'coo_sub': coo_sub,
        'coo_save': coo_save,
        'coo_before_anneal': coo_before_anneal,
        'coo_final': coo_final
    }

    return result


def main():
    if len(sys.argv) < 3:
        print('Usage: python generate_atom_info.py <src> <n_images>')
        sys.exit(1)

    src = sys.argv[1]
    n_incidence = int(sys.argv[2])

    idx_range = range(n_incidence + 1)
    save_freq = 1000
    my_atom_dict = AtomDict()

    for struct_idx in idx_range:
        coo_dict = load_coo(src, struct_idx)
        my_atom_dict.update(coo_dict, struct_idx)

        if struct_idx > 0 and struct_idx % save_freq == 0:
            print(f'Saving the current state at structure {struct_idx}')
            # with open(f'atom_dict_{struct_idx}.pkl', 'wb') as f:
            #     pickle.dump(my_atom_dict, f)
            my_atom_dict.atomdict_to_dataframe(struct_idx)
    # with open(f'atom_dict_final.pkl', 'wb') as f:
    #     pickle.dump(my_atom_dict, f)
    my_atom_dict.atomdict_to_dataframe(n_incidence)


if __name__ == '__main__':
    main()
