import numpy as np
from ase.io import read
import ase.build

from collections import Counter

class UniqueRxnGenerator:
    def run(self):
        pass

def get_stoichiometry(chemical_symbols, symbol_order):
    species_dict = dict(Counter(chemical_symbols))
    result = []
    for symbol in symbol_order:
        result.append(species_dict.get(symbol, 0))

    return np.array(result)


def helper(i, comp_change, atom_name, unique_bulk_idx):
    a = unique_bulk_idx
    gas_dict = {
        "HF": np.array([0,0,0,1,1], dtype=int),
        "O2": np.array([0,2,0,0,0], dtype=int),
        "SiF2": np.array([1,0,0,0,2], dtype=int),
        "SiF4": np.array([1,0,0,0,4], dtype=int),
        "CO": np.array([0,1,1,0,0], dtype=int),
        "CF": np.array([0,0,1,0,1], dtype=int),
        "CO2": np.array([0,2,1,0,0], dtype=int),
    }

    if np.sum(np.abs(comp_change))==0:
        line = f"{a[i-1,0]}_{a[i-1,1]}/{a[i,0]}_{a[i,1]}\n"
        return line

    gas = None
    for key, value in gas_dict.items():
        if np.all(np.abs(comp_change)==value):
            gas = key
            break

    if gas is not None:
        if np.sum(np.abs(comp_change)) == 1:
            gas = atom_name[np.nonzero(comp_change)[0][0]]

        if np.sum(comp_change) > 0:
            line = f"{a[i-1,0]}_{a[i-1,1]},{gas}/{a[i,0]}_{a[i,1]}\n"
            return line
        else:
            line = f"{a[i-1,0]}_{a[i-1,1]}/{a[i,0]}_{a[i,1]},{gas}\n"
            return line
    else:
        line = f"{a[i-1,0]}_{a[i-1,1]}/{a[i,0]}_{a[i,1]}/{comp_change}, needs change!\n"
        return line


def main():
    file_name = "unique_bulk.dat"

    with open(file_name, 'r') as O:
        unique_bulk_idx = np.array([
            [int(i) for i in line.split()[0].split(",")]
            for line in O.readlines()[1:]
        ])

    path_dump = "unique_bulk.extxyz"
    dump = read(path_dump, index=":",format='extxyz')

    with open("desorbed_gas_id.dat",'r') as O:
        lines=[j.split("/") for j  in O.readlines()[1:]]
        for i in range(len(lines)):
            incidence, gas_idx, composition, *_ = lines[i]
            lines[i]=[incidence, gas_idx]+composition.split()

    with open("rxn.dat",'w') as O:
        O.write("#reactants,/products,/add_to_rxn_E\n")

    atom_name = ["Si", "O", "C", "H", "F"]
    for i in range(1,len(dump)):
        composition0 =  get_stoichiometry(dump[i-1].get_chemical_symbols(), atom_name)
        composition1 =  get_stoichiometry(dump[i].get_chemical_symbols(), atom_name)
        comp_change = composition1 - composition0

        with open("rxn.dat",'a') as O:
            line = helper(i, comp_change, atom_name, unique_bulk_idx)
            O.write(line)


if __name__ == "__main__":
    main()
