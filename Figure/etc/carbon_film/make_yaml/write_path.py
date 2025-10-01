import os
import yaml


def main():
    path_list = [
        "/data_etch/data_HM/nurion/set_1/",
        "/data_etch/data_HM/nurion/set_2/",
        "/data_etch/data_HM/nurion/set_3/",
        "/data_etch/data_HM/nurion/set_uncomplete/",

        "/data2_1/andynn/Etch/data_nurion/set_3/",
        "/data2_1/andynn/Etch/data_nurion/set_uncomplete/",
    ]
    result = {}
    for path in path_list:
        for file in os.listdir(path):
            if not file.endswith('_coo.tar.gz'):
                continue

            full_path = os.path.join(path, str(file).replace('.tar.gz', ''))
            try:
                ion_type, ion_energy = file.replace('_coo.tar.gz', '').split('_')
                ion_energy = int(ion_energy)
                if ion_type not in result:
                    result[ion_type] = {}
                result[ion_type][ion_energy] = full_path
            except:
                breakpoint()

    with open('path.yaml', 'w') as f:
        yaml.dump(result, f)


if __name__ == "__main__":
    main()
