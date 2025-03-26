import sys
from collections import defaultdict
from dataclasses import dataclass
import pickle


@dataclass
class SiO2ConvDict:
    conv_dict = {
            'F4Si1': (1, 0, 0, 0, 4),
            'F3H1Si1': (1, 0, 0, 1, 3),
            'F2H2Si1': (1, 0, 0, 2, 2),
            'F1H3Si1': (1, 0, 0, 3, 1),
            'H4Si1': (1, 0, 0, 4, 0),

            'F2Si1': (1, 0, 0, 0, 2),

            'O2': (0, 2, 0, 0, 0),
            'H2': (0, 0, 0, 2, 0),
            'F2': (0, 0, 0, 0, 2),
            'C1O1': (0, 1, 1, 0, 0),
            'F1H1': (0, 0, 0, 1, 1),

            'C1F4': (0, 0, 1, 0, 4),
            'C1F3H1': (0, 0, 1, 1, 3),
            'C1F2H2': (0, 0, 1, 2, 2),
            'C1F1H3': (0, 0, 1, 3, 1),
            'C1H4': (0, 0, 1, 4, 0),

            'C1F2': (0, 0, 1, 0, 2),

            'H2O1': (0, 1, 0, 2, 0),
            'F2O1': (0, 1, 0, 0, 2),
            'F1H1O1': (0, 1, 0, 1, 1),

            'C1H2O1': (0, 1, 1, 2, 0),
            'C1F1H1O1': (0, 1, 1, 1, 1),
            'C1F2O1': (0, 1, 1, 0, 2),
            'C1O2': (0, 2, 1, 0, 0),
            }


@dataclass
class Si3N4ConvDict:
    conv_dict = {
            'F4Si1': (1, 0, 0, 0, 4),
            'F3H1Si1': (1, 0, 0, 1, 3),
            'F2H2Si1': (1, 0, 0, 2, 2),
            'F1H3Si1': (1, 0, 0, 3, 1),
            'H4Si1': (1, 0, 0, 4, 0),

            'F2Si1': (1, 0, 0, 0, 2),

            'N2': (0, 2, 0, 0, 0),
            'H2': (0, 0, 0, 2, 0),
            'F2': (0, 0, 0, 0, 2),
            'C1N1': (0, 1, 1, 0, 0),
            'F1H1': (0, 0, 0, 1, 1),
            'H3N1': (0, 1, 0, 2, 0),
            'C1F1N1': (0, 1, 1, 0, 1),
            'C1H1N1': (0, 1, 1, 1, 0),

            'C1F4': (0, 0, 1, 0, 4),
            'C1HF3': (0, 0, 1, 1, 3),
            'C1H2F2': (0, 0, 1, 2, 2),
            'C1H3F1': (0, 0, 1, 3, 1),
            'C1H4': (0, 0, 1, 4, 0),

            'C1F2': (0, 0, 1, 0, 2),
            }



def read_files(src, mol_dict):
    with open(src, "r") as f:
        lines = f.readlines()

    read_cluster = False
    incidence_count = 0

    for line in lines:
        if 'time spent' in line:
            read_cluster = True
            incidence_count = int(line.split()[1])
            continue

        if read_cluster:
            if line.startswith('Cluster'):
                mol_type = line.split()[3]
                is_in_list = line.split()[-1] == 'True'
                if is_in_list:
                    mol_dict[mol_type].append(incidence_count)
                    # print(mol_type, incidence_count)
                continue
            else:
                read_cluster = False
                continue


def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_mol_dict.py <path/to/cluster.log> ...")
        sys.exit(1)
    src_files = sys.argv[1:]

    mol_dict = defaultdict(list)
    for src in src_files:
        read_files(src, mol_dict)

    conv_dict = SiO2ConvDict().conv_dict
    # conv_dict = Si3N4ConvDict().conv_dict

    mol_dict = {conv_dict[k]: v for k, v in mol_dict.items()}
    with open("mol_dict.pkl", "wb") as f:
        pickle.dump(mol_dict, f)


if __name__ == "__main__":
    main()
