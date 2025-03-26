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
    @staticmethod
    def run(file):
        if FileNameChecker.is_initial_coo(file):
            return 'initial'
        elif FileNameChecker.is_rm_byproduct(file):
            return 'rm_byproduct'
        elif FileNameChecker.is_add(file):
            return 'add'
        elif FileNameChecker.is_before_anneal(file):
            return 'before_anneal'
        elif FileNameChecker.is_sub(file):
            return 'sub'
        elif FileNameChecker.is_save(file):
            return 'save'
        elif FileNameChecker.is_final(file):
            return 'final'
        else:
            return None
    @staticmethod
    def extract_index(file):
        match = re.search(r'(\d+)', file)
        return int(match.group(1)) if match else None


def list_up_coo_files(path):
    """
    Scans the given directory and groups files by their numeric index.
    Returns a dictionary where the keys are indices and values are lists of files.
    """
    grouped_files = defaultdict(list)
    for root, _, files in os.walk(path):
        for file in files:
            file_type = FileNameChecker.run(file)
            if file_type is None:
                continue
            index = FileNameChecker.extract_index(file)
            if index is None:
                continue
            grouped_files[index].append((index, file_type, os.path.join(root, file)))
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


def process_batch(start_idx, end_idx, files, path_cutoff):
    """
    Processes each batch and saves the results.
    If the output file already exists, it loads and returns the existing data.
    """
    # filename = f"{start_idx:04d}_to_{end_idx:04d}.pkl"
    filename = f"{start_idx:04d}_to_{end_idx:04d}.h5"

    if os.path.exists(filename):
        print(f"Skipping {filename} (Already exists)")
        return

    result = nested_dict()
    processor = BondDataGenerator(path_cutoff)
    for (struct_idx, file_type, file) in files:
        bonding_data = processor.run(file)
        print(f"{file} done")
        result[struct_idx][file_type] = bonding_data

    with open(filename, 'wb') as f:
        pickle.dump(result, f)
    bonding_result_to_df(result, filename)

    print(f"✅ Saved: {filename}")


def get_data(my_path, path_cutoff):
    """
    Main function to scan directory, group files, create batches, and process them sequentially.
    """
    grouped_files = list_up_coo_files(my_path)
    # batches = get_batches_from_groups(grouped_files, batch_size=1000)
    batches = get_batches_from_groups(grouped_files, batch_size=1000)

    total_batches = len(batches)
    for batch_num, (start_idx, end_idx, batch_files) in enumerate(batches, start=1):
        process_batch(start_idx, end_idx, batch_files, path_cutoff)
        print(f"Batch {batch_num}/{total_batches} done")


def main():
    if len(sys.argv) < 5:
        print("Usage: python generate_bondinfo.py <path_yaml> <path_cutoff> <ion_type> <ion_E>")
        sys.exit()
    path_yaml = sys.argv[1]
    path_cutoff = sys.argv[2]
    ion_type = sys.argv[3]
    ion_E = int(sys.argv[4])
    with open(path_yaml, 'r') as f:
        paths = yaml.safe_load(f)
    my_path = paths[ion_type][ion_E]
    get_data(my_path, path_cutoff)


if __name__ == '__main__':
    main()
