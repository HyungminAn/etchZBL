import sys
import pickle
import time
import numpy as np
import pandas as pd

from AtomInfo import AtomInfo, AtomDict, nested_dict
from BondData import BondingData
from utils import timeit


class CarbonStateClassifier:
    def __init__(self, atom_df, bond_df):
        self.atom_df = atom_df
        self.bond_df = bond_df

        self.created_dict, self.removed_dict = self._build_timestamp_dicts()
        self.result_cache = {}

    def _build_timestamp_dicts(self):
        df = self.atom_df.reset_index()
        grouped = df.groupby('global_idx')
        created = grouped['timestamp_created'].min().to_dict()
        removed = grouped['timestamp_removed'].apply(
            lambda x: int(x.dropna().min()) if not x.isna().all() else None
        ).to_dict()
        return created, removed

    def get_atom_row_by_global(self, global_idx, struct_idx, file_type):
        key = (global_idx, struct_idx, file_type)
        if key in self.atom_df.index:
            return self.atom_df.loc[key]
        else:
            return None

    def get_local_idx_by_global(self, global_idx, struct_idx, file_type):
        row = self.get_atom_row_by_global(global_idx, struct_idx, file_type)
        return row['local_idx'] if row is not None else None

    def get_neighbor_symbols(self, struct_idx, file_type, carbon_index):
        key = (struct_idx, file_type, carbon_index)
        if key in self.bond_df.index:
            return self.bond_df.loc[(struct_idx, file_type, carbon_index)]['neighbor_symbols']
        else:
            return None

    def build_result_cache(self, final_struct_idx, n_carbon):
        for global_idx in range(n_carbon):
            self.run(global_idx, final_struct_idx)

    def run(self, global_idx, struct_idx):
        if global_idx >= struct_idx:
            return "NOT_CREATED"

        if self.created_dict.get(global_idx) is None:
            return "REFLECTED"

        cache = False
        atom_removed = self.removed_dict.get(global_idx)
        if atom_removed is not None and atom_removed <= struct_idx:
            cache = True
            cache_result = self.result_cache.get(global_idx)
            if cache_result is not None:
                return cache_result

            key = (global_idx, atom_removed, 'initial')
            if key not in self.atom_df.index:
                self.result_cache[global_idx] = "REMOVED_DURING_MD"
                return "REMOVED_DURING_MD"

            key = (global_idx, atom_removed, 'rm_byproduct')
            if key in self.atom_df.index:
                ref_filetype, ref_idx, prefix = 'rm_byproduct', atom_removed, ''
            else:
                ref_filetype, ref_idx, prefix = 'initial', atom_removed, 'BYPRODUCT_'
        else:
            ref_filetype, ref_idx, prefix = 'initial', struct_idx, ''

        key = (global_idx, ref_idx, ref_filetype)
        if key not in self.atom_df.index:
            return "REFLECTED"

        local_idx = self.atom_df.loc[(global_idx, ref_idx, ref_filetype)]['local_idx']
        neighbor_symbols = self.get_neighbor_symbols(ref_idx, ref_filetype, local_idx)
        state_C = self.get_bondtype(neighbor_symbols)
        if struct_idx == 300:
            print(f"global_idx: {global_idx}, neighbor_symbols: {neighbor_symbols}, state_C: {state_C}")
        result = f'{prefix}{state_C}' if prefix != 'BYPRODUCT_' else 'BYPRODUCT'
        if cache:
            self.result_cache[global_idx] = result
        return result

    @staticmethod
    def get_bondtype(symbols):
        symbols = symbols.split()
        symbol_set = set(symbols)
        if symbol_set == {'C'}:
            return f"C{symbols.count('C')}"
        elif symbol_set == {'Si', 'C'}:
            return "SiC_cluster"
        elif 'F' in symbols:
            return "Fluorocarbon"
        elif 'O' in symbols:
            return "with_O"
        else:
            return "etc"


class CarbonFilter:
    def __init__(self, atom_df, bond_df, n_struct, n_carbon):
        self.atom_df = atom_df
        self.bond_df = bond_df
        self.n_struct = n_struct
        self.n_carbon = n_carbon
        self.exclude_dict = self._generate_exclude_dict()

        self.call_count = 0
        self.start_time = time.perf_counter()
        self.end_time = time.perf_counter()

    def check(self, global_idx):
        return self.exclude_dict.get(global_idx, False)

    @timeit
    def _generate_exclude_dict(self):
        classifier = CarbonStateClassifier(self.atom_df, self.bond_df)
        exclude_list = ['NOT_CREATED']
        exclude_dict = {}

        initial_atoms = self.atom_df.reset_index()
        final_atoms = initial_atoms[(initial_atoms['struct_idx'] == self.n_struct) & (initial_atoms['file_type'] == 'initial')]

        for _, row in final_atoms.iterrows():
            global_idx = row['global_idx']
            state = classifier.run(global_idx, self.n_struct)
            exclude_dict[global_idx] = any(ex in state for ex in exclude_list)

        return exclude_dict


class DataCombinator:
    def __init__(self, path_atom_data, path_bond_data, coo_stat):
        self.path_atom_data = path_atom_data
        self.path_bond_data = path_bond_data
        self.coo_stat = coo_stat

    @timeit
    def load_dict(self):
        coo_stat = self.coo_stat

        start_idx = coo_stat['start_idx']
        end_idx = coo_stat['end_idx']
        stride = coo_stat['stride']
        count = coo_stat['count']
        n_carbon = coo_stat['n_carbon']

        path_atom_data = f'{self.path_atom_data}/atom_dict_{n_carbon}.h5'
        df_atom = pd.read_hdf(path_atom_data, key='atom_dict')
        print(f"Loaded {path_atom_data}")

        # bond_data = {}
        df_list = []
        for _ in range(count):
            path_bond_data = f'{self.path_bond_data}/{start_idx:04}_to_{end_idx:04}.h5'
            df = pd.read_hdf(path_bond_data, key='bond')
            df_list.append(df)
            if start_idx == 0:
                start_idx += 1
            start_idx += stride
            end_idx += stride
            print(f"Loaded {path_bond_data}")

        df_bond = pd.concat(df_list, ignore_index=True)
        print("Loaded data")

        df_atom = df_atom.set_index(['global_idx', 'struct_idx', 'file_type'])
        df_bond = df_bond.set_index(['struct_idx', 'file_type', 'carbon_index'])

        return df_atom, df_bond

    @timeit
    def run(self):
        coo_stat = self.coo_stat

        coo_stat['n_struct'] = coo_stat['end_idx'] * coo_stat['count']
        coo_stat['n_carbon'] = coo_stat['n_struct']

        n_struct = coo_stat['n_struct']
        n_carbon = coo_stat['n_carbon']

        df_atom, df_bond = self.load_dict()
        cf = CarbonFilter(df_atom, df_bond, n_struct, n_carbon)
        classifier = CarbonStateClassifier(df_atom, df_bond)
        classifier.build_result_cache(n_struct, n_carbon)

        # records = []
        records_np = np.empty((n_struct * n_carbon,), dtype=[
            ('struct_idx', 'i4'),
            ('global_idx', 'i4'),
            ('state_C', 'U20')
        ])
        start_time = time.perf_counter()
        log_step = 100

        np_idx = 0

        for struct_idx in range(n_struct):
            for global_idx in range(n_carbon):
                if cf.check(global_idx):
                    continue
                state_C = classifier.run(global_idx, struct_idx)

                records_np[np_idx] = (struct_idx, global_idx, state_C)
                np_idx += 1

            if struct_idx % log_step == 0:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Processed {struct_idx}/{n_struct} structures, elapsed time: {elapsed_time:.2f} sec")
                start_time = end_time

        records = records_np[:np_idx]
        df = pd.DataFrame.from_records(records)
        H5_SAVE_OPTS = {
            'key': 'df',
            'mode': 'w',
            'format': 'table',
            'complevel': 9,
            'complib': 'blosc:zstd',
            'index': False,
        }
        df.to_hdf('total_dict.h5', **H5_SAVE_OPTS)


def main():
    src_atom_data = sys.argv[1]
    src_bond_data = sys.argv[2]
    n_incidence = sys.argv[3]

    batch_count = 1000
    count = int(n_incidence) // batch_count

    coo_stat = {
        'start_idx': 0,
        'end_idx': batch_count,
        'stride': batch_count,
        'count': count,
    }

    combinator = DataCombinator(src_atom_data, src_bond_data, coo_stat)
    combinator.run()


if __name__ == '__main__':
    main()
