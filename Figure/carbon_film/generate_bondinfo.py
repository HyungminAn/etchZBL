import os
import pickle
import sys
import re
from collections import defaultdict
import pandas as pd

import yaml
from utils import timeit
from BondData import BondDataGenerator
from AtomInfo import nested_dict

class FileNameChecker:
    @staticmethod
    def is_initial_coo(file):
        return re.compile(r'^str_shoot_\d+\.coo$').match(file)
    @staticmethod
    def is_rm_byproduct(file):
        return re.compile(r'^rm_byproduct_str_shoot_\d+\.coo$').match(file)
    @staticmethod
    def is_add(file):
        return re.compile(r'^add_str_shoot_\d+\.coo$').match(file)
    @staticmethod
    def is_before_anneal(file):
        return re.compile(r'^str_shoot_\d+\_after_mod_before_anneal.coo$').match(file)
    @staticmethod
    def is_sub(file):
        return re.compile(r'^sub_str_shoot_\d+\.coo$').match(file)
    @staticmethod
    def is_save(file):
        return re.compile(r'^save_str_shoot_\d+\.coo$').match(file)
    @staticmethod
    def is_final(file):
        return re.compile(r'^str_shoot_\d+\_after_mod.coo$').match(file)
    def run(self, file):
        if self.is_initial_coo(file):
            return 'initial'
        elif self.is_rm_byproduct(file):
            return 'rm_byproduct'
        elif self.is_add(file):
            return 'add'
        elif self.is_before_anneal(file):
            return 'before_anneal'
        elif self.is_sub(file):
            return 'sub'
        elif self.is_save(file):
            return 'save'
        elif self.is_final(file):
            return 'final'
        else:
            return None
    @staticmethod
    def extract_index(file):
        match = re.search(r'(\d+)', file)
        return int(match.group(1)) if match else None

class FileNameCheckerReduced(FileNameChecker):
    def run(self, file):
        if self.is_rm_byproduct(file):
            return 'rm_byproduct'
        else:
            return None

@timeit
def list_up_coo_files(path_list, fast_mode=False):
    """
    Scans the given directory and groups files by their numeric index.
    Returns a dictionary where the keys are indices and values are lists of files.
    """
    grouped_files = defaultdict(list)
    if fast_mode:
        file_checker = FileNameCheckerReduced()
    else:
        file_checker = FileNameChecker()

    for path in path_list:
        for root, _, files in os.walk(path):
            for file in files:
                print(file)
                file_type = file_checker.run(file)
                if file_type is None:
                    continue
                index = file_checker.extract_index(file)
                if index is None:
                    continue
                value = (index, file_type, os.path.join(root, file))
                if value in grouped_files[index]:
                    continue
                grouped_files[index].append(value)
    return grouped_files


def get_batches_from_groups(grouped_files, batch_size=100):
    """
    Creates batches based on grouped file indices.
    Ensures that files with the same index are processed together.
    """
    sorted_indices = sorted(grouped_files.keys())
    batches = []
    current_batch = []
    start_idx = None

    for index in sorted_indices:
        if start_idx is None:
            start_idx = index
        current_batch.extend(grouped_files[index])

        if index > 0 and index % batch_size == 0:
            batches.append((start_idx, index, current_batch))
            current_batch = []
            start_idx = None  # Reset for new batch

    if current_batch:
        batches.append((start_idx, index, current_batch))

    return batches


def bonding_result_to_df(result, filename):
    records = []
    for struct_idx, filetypes in result.items():
        for file_type, bond_data in filetypes.items():
            for info in bond_data.carbon_neighbors_info:
                records.append({
                    "struct_idx": struct_idx,
                    "file_type": file_type,
                    "carbon_index": info["carbon_index"],
                    "neighbor_symbols": info["neighbor_symbols"],
                })
    df = pd.DataFrame(records)
    df['neighbor_symbols'] = df['neighbor_symbols'].apply(lambda x: ' '.join(x))

    H5_SAVE_OPTS = {
        'key': 'bond',
        'mode': 'w',
        'format': 'table',
        'complevel': 9,
        'complib': 'blosc:zstd',
        'index': False,
    }
    df.to_hdf(filename, **H5_SAVE_OPTS)


def process_batch(start_idx, end_idx, files, path_cutoff, dst, step=10):
    """
    Processes each batch and saves the results.
    If the output file already exists, it loads and returns the existing data.
    """
    # filename = f"{start_idx:04d}_to_{end_idx:04d}.pkl"
    filename = f"{dst}/"
    if start_idx >= 10000:
        filename += f"{start_idx:05d}_to_"
    else:
        filename += f"{start_idx:04d}_to_"

    if end_idx >= 10000:
        filename += f"{end_idx:05d}.h5"
    else:
        filename += f"{end_idx:04d}.h5"

    if os.path.exists(filename):
        print(f"Skipping {filename} (Already exists)")
        return

    result = nested_dict()
    processor = BondDataGenerator(path_cutoff)
    for (struct_idx, file_type, file) in files[::step]:
        bonding_data = processor.run(file)
        print(f"{file} done")
        result[struct_idx][file_type] = bonding_data

    # with open(filename, 'wb') as f:
    #     pickle.dump(result, f)
    bonding_result_to_df(result, filename)

    print(f"✅ Saved: {filename}")


def get_data(my_path_list, path_cutoff, dst):
    """
    Main function to scan directory, group files, create batches, and process them sequentially.
    """
    grouped_files = list_up_coo_files(my_path_list, fast_mode=True)
    batches = get_batches_from_groups(grouped_files, batch_size=100000)
    total_batches = len(batches)

    for batch_num, (start_idx, end_idx, batch_files) in enumerate(batches, start=1):
        process_batch(start_idx, end_idx, batch_files, path_cutoff, dst)
        print(f"Batch {batch_num}/{total_batches} done")


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_bondinfo.py <path_yaml> <path_cutoff>")
        sys.exit()

    path_yaml, path_cutoff = sys.argv[1:]
    with open(path_yaml, 'r') as f:
        paths = yaml.safe_load(f)
    for ion_type, energies in paths.items():
        for ion_E, my_path_list in energies.items():
            dst = f"{ion_type}_{ion_E}"
            os.makedirs(dst, exist_ok=True)
            get_data(my_path_list, path_cutoff, dst)


if __name__ == '__main__':
    main()
