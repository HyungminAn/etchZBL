import pickle
import time
import pandas as pd

from AtomInfo import AtomInfo, AtomDict, nested_dict
from BondData import BondingData
from utils import timeit


class CarbonStateClassifier:
    def __init__(self, atom_dict, bond_data):
        self.atom_dict = atom_dict
        self.bond_data = bond_data

    def run(self, atom, global_idx, struct_idx):
        if global_idx >= struct_idx:
            return "NOT_CREATED"

        if atom is None:
            return "REFLECTED"

        removed_idx = atom.timestamp_removed
        if removed_idx is None or removed_idx > struct_idx:
            ref_filetype, ref_idx, prefix = 'initial', struct_idx, ''
        elif removed_idx <= struct_idx:
            if self.atom_dict.map_global_to_local[global_idx][removed_idx].get('initial') is None:
                return "REMOVED_DURING_MD"
            elif self.atom_dict.map_global_to_local[global_idx][removed_idx].get('rm_byproduct') is None:
                ref_filetype, ref_idx, prefix = 'initial', removed_idx, 'BYPRODUCT_'
            elif self.atom_dict.map_global_to_local[global_idx][removed_idx].get('sub') is None:
                ref_filetype, ref_idx, prefix = 'rm_byproduct', removed_idx, ''
            else:
                raise NotImplementedError('This should not happen')
        else:
            raise NotImplementedError('This should not happen')

        local_idx = self.atom_dict.map_global_to_local[global_idx][ref_idx][ref_filetype]
        ref_filetype = 'normal' if ref_filetype == 'initial' else ref_filetype
        neigh_info = self.bond_data[ref_idx][ref_filetype].carbon_neighbors_info
        state_C = self.get_bondtype(self.get_neighbor_symbols(neigh_info, local_idx))

        if prefix == 'BYPRODUCT_':
            return 'BYPRODUCT'
        else:
            return f'{prefix}{state_C}'


    @staticmethod
    def get_bondtype(symbols):
        symbol_set = set(symbols)
        if symbol_set == {'C'}:
            n_C = symbols.count('C')
            return f"C{n_C}"
        elif symbol_set == {'Si', 'C'}:
            return "SiC_cluster"
        elif 'F' in symbols:
            return "Fluorocarbon"
        elif 'O' in symbols:
            return "with_O"
        else:
            return "etc"

        '''sp with C and Si
        n_Si = symbols.count('Si')
        n_C = symbols.count('C')
        n_neighbor = len(symbols)
        name = ""
        if n_neighbor > 5:
            name = "Etc"
        elif n_neighbor == 4:
            name = "sp3"
        elif n_neighbor == 3:
            name = "sp2"
        elif n_neighbor == 2:
            name = "sp"
        else:
            name = "Etc"
        if name.startswith("sp"):
            name += '_'
            if n_C > 0:
                name += f"C{n_C}"
            if n_Si > 0:
                name += f"Si{n_Si}"
        return name
        '''

        '''simple case
        elem_list = ['C', 'Si', 'O', 'H', 'F']
        name = ""
        for elem in elem_list:
            count = symbols.count(elem)
            if count > 0:
                name += f"{elem}{count}"
        return name
        '''

        '''case 2
        n_C = symbols.count('C')
        if n_C >= 2:
            return 'CF_film'

        n_Si = symbols.count('Si')
        if n_Si == 0:
            return 'COxFy'

        elem_list = ['Si', 'O', 'H', 'F']
        name = ""
        for elem in elem_list:
            count = symbols.count(elem)
            if count > 0:
                name += f"{elem}{count}"

        return name
        '''

        '''case 1
        name = ""
        n_C = symbols.count('C')
        n_Si = symbols.count('Si')
        if n_C + n_Si >= 2:
            if n_C > 0:
                name += f"C{n_C}"
            if n_Si > 0:
                name += f"Si{n_Si}"
        else:
            name += "others"
        return name
        '''

    @staticmethod
    def get_neighbor_symbols(lst, val):
        return next((d['neighbor_symbols'] for d in lst if d['carbon_index'] == val), None)


class CarbonFilter:
    def __init__(self, atom_dict, bond_data, n_struct, n_carbon):
        self.atom_dict = atom_dict
        self.bond_data = bond_data
        self.n_struct = n_struct
        self.n_carbon = n_carbon

        self.exclude_dict = self._generate_exclude_dict()

    def check(self, global_idx):
        return self.exclude_dict[global_idx]

    def _generate_exclude_dict(self):
        classifier = CarbonStateClassifier(self.atom_dict, self.bond_data)
        # exclude_list = ['NOT_CREATED',
        #                 'REFLECTED',
        #                 'REMOVED_DURING_MD',
        #                 'BYPRODUCT_']
        exclude_list = ['NOT_CREATED']
        exclude_dict = {key: False for key in range(self.n_carbon)}
        valid_atoms = self.atom_dict.map_local_to_global[self.n_struct]['initial']
        for local_idx, atom in valid_atoms.items():
            exclude_this = False
            state_C = classifier.run(atom, atom.global_idx, self.n_struct)
            # print(f"Checking {atom.global_idx} due to {state_C}")
            for exclude in exclude_list:
                if exclude in state_C:
                    exclude_this = True
                    # print(f"Excluding {atom.global_idx} due to {state_C}")
                    break
            exclude_dict[atom.global_idx] = exclude_this
        return exclude_dict


class DataCombinator:
    def __init__(self):
        pass

    @timeit
    def load_dict(self, coo_stat):
        start_idx = coo_stat['start_idx']
        end_idx = coo_stat['end_idx']
        stride = coo_stat['stride']
        count = coo_stat['count']
        n_carbon = coo_stat['n_carbon']

        path_atom_data = f'../../01_GenAtomInfo/gen_AtomInfo/atom_dict_{n_carbon}.pkl'
        with open(path_atom_data, 'rb') as f:
            atom_dict = pickle.load(f)
        print(f"Loaded {path_atom_data}")

        bond_data = {}
        for _ in range(count):
            path_bond_data = f'../../02_GenBondData/save_by_1000/{start_idx:04}_to_{end_idx:04}.pkl'
            with open(path_bond_data, 'rb') as f:
                my_dict = pickle.load(f)
                bond_data.update(my_dict)
            if start_idx == 0:
                start_idx += 1
            start_idx += stride
            end_idx += stride
            print(f"Loaded {path_bond_data}")
        print("Loaded data")

        return atom_dict, bond_data

    @timeit
    def count_global_idx(self, atom_dict, cf, filter_stable, n_struct):
        result = {}
        file_type_list = ['initial',
                          'rm_byproduct',
                          'add',
                          'before_anneal',
                          'sub',
                          'save',
                          'final']

        for struct_idx in range(n_struct+1):
            for file_type in file_type_list:
                atom_list = atom_dict.map_local_to_global[struct_idx][file_type]

                for local_idx, atom in atom_list.items():
                    if filter_stable and cf.check(atom.global_idx):
                        continue

                    global_idx = atom.global_idx
                    if result.get(global_idx) is not None:
                        continue

                    result[global_idx] = atom
        return result

    @timeit
    def make_state_dict(self,
                        atom_dict,
                        bond_data,
                        cf,
                        global_idx_dict,
                        n_struct,
                        n_carbon,
                        log_step=100,
                        ):
        total_dict = nested_dict()
        classifier = CarbonStateClassifier(atom_dict, bond_data)

        start_time = time.perf_counter()
        for struct_idx in range(n_struct):
            for global_idx in range(n_carbon):
                if cf.check(global_idx):
                    continue
                atom = global_idx_dict.get(global_idx)
                state_C = classifier.run(atom, global_idx, struct_idx)
                total_dict[struct_idx][global_idx] = state_C
            if struct_idx % log_step == 0:
                end_time = time.perf_counter()
                time_elapsed = end_time - start_time
                print(f"Processed {struct_idx}/{n_struct}; Time elapsed: {time_elapsed:.2f} s")
                start_time = end_time
        return total_dict

    @timeit
    def make_dict(self, coo_stat, filter_stable=False):
        n_struct = coo_stat['n_struct']
        n_carbon = coo_stat['n_carbon']

        atom_dict, bond_data = self.load_dict(coo_stat)
        cf = CarbonFilter(atom_dict,
                          bond_data,
                          n_struct,
                          n_carbon)
        global_idx_dict = self.count_global_idx(atom_dict,
                                                cf,
                                                filter_stable,
                                                n_struct)
        total_dict = self.make_state_dict(atom_dict,
                                          bond_data,
                                          cf,
                                          global_idx_dict,
                                          n_struct,
                                          n_carbon)
        with open('total_dict.pkl', 'wb') as f:
            pickle.dump(total_dict, f)

    @timeit
    def run(self):
        coo_stat = {
            'start_idx': 0,
            'end_idx': 1000,
            'stride': 1000,
            'count': 1,
        }
        coo_stat['n_struct'] = coo_stat['end_idx'] * coo_stat['count']
        coo_stat['n_carbon'] = coo_stat['n_struct']

        # self.make_dict(coo_stat, filter_stable=True)
        self.make_dict(coo_stat, filter_stable=False)

def main():
    combinator = DataCombinator()
    combinator.run()


if __name__ == '__main__':
    main()
