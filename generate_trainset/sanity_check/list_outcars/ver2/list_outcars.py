import os
from collections import defaultdict


def main():
    with open('list', 'r') as f:
        lines = f.readlines()
        folder_list = [line.strip() for line in lines]

    outcar_dict = defaultdict(list)
    for folder in folder_list:
        # Check recursively all subfolders for OUTCAR files
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file == 'OUTCAR':
                    file_name = os.path.join(root, file)
                    outcar_dict[folder].append(file_name)
        print(f'Folder {folder} has {len(outcar_dict[folder])} OUTCAR files')

    with open('outcars', 'w') as f:
        for folder in folder_list:
            for outcar in outcar_dict[folder]:
                f.write(outcar + '\n')


if __name__ == '__main__':
    main()
