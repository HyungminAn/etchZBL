import os


def main():
    src = "./poscars/"
    file_list = [os.path.join(src, i) for i in os.listdir(src)]

    gas_dict = {}
    for file in file_list:
        gas_type = file.split("_")[-1]
        if gas_type not in gas_dict:
            gas_dict[gas_type] = []
        gas_dict[gas_type].append(file)

    os.makedirs("./poscars_classified", exist_ok=True)
    exclude_list = [
        "C",
        "H",
        "O",
        "F",
        "CO",
        "O2",
        "HF",
        ]
    for key, value in gas_dict.items():
        if key in exclude_list:
            continue
        print(key, len(value))

        os.makedirs(f"./poscars_classified/{key}", exist_ok=True)
        for file in value:
            os.system(f"cp {file} ./poscars_classified/{key}")



if __name__ == "__main__":
    main()
