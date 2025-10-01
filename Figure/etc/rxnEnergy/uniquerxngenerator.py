from collections import Counter

import numpy as np
from ase.io import read

from utils import PARAMS

class UniqueRxnGenerator:
    def run(self):
        print("Running UniqueRxnGenerator...")
        unique_bulk_idx = self.get_bulk_indices()
        dump = read(PARAMS.path_unique_bulk_extxyz, index=":", format='extxyz')
        line = self.get_reaction_data(dump, unique_bulk_idx)
        with open(PARAMS.path_reaction_data, 'w') as f:
            f.write(line)
        print("Unique reactions generated successfully.")

    def get_bulk_indices(self):
        with open(PARAMS.path_unique_bulk, 'r') as f:
            lines = f.readlines()[1:]
        result = []
        for line in lines:
            inc, img = line.split(",")
            result.append([int(inc), int(img)])
        return np.array(result)

    def get_stoichiometry(self, chemical_symbols):
        species_dict = dict(Counter(chemical_symbols))
        result = []
        for symbol in PARAMS.SYSTEM_DEPENDENT.ELEM_LIST:
            result.append(species_dict.get(symbol, 0))
        return np.array(result)

    def get_reaction_data(self, dump, unique_bulk_idx):
        line = "#reactants,/products,/add_to_rxn_E\n"
        atom_name = PARAMS.SYSTEM_DEPENDENT.ELEM_LIST

        for i, (image0, image1) in enumerate(zip(dump[:-1], dump[1:]), start=1):
            inc0, img0 = unique_bulk_idx[i-1]
            inc1, img1 = unique_bulk_idx[i]
            rxn0 = f"{inc0}_{img0}"
            rxn1 = f"{inc1}_{img1}"

            comp0 = self.get_stoichiometry(image0.get_chemical_symbols())
            comp1 = self.get_stoichiometry(image1.get_chemical_symbols())
            comp_change = comp1 - comp0

            if np.all(comp_change == 0):
                line += f"{rxn0}/{rxn1}\n"
                continue

            neg_vec = np.where(comp_change < 0, -comp_change, 0)
            pos_vec = np.where(comp_change > 0,  comp_change, 0)

            removed = None
            added = None

            for gas, vec in PARAMS.SYSTEM_DEPENDENT.gas_dict.items():
                vec = np.array(vec)
                if removed is None and np.array_equal(neg_vec, vec):
                    removed = gas
                if added is None and np.array_equal(pos_vec, vec):
                    added = gas

            if removed is None and neg_vec.sum() == 1:
                idx = np.nonzero(neg_vec)[0][0]
                removed = atom_name[idx]
            if added is None and pos_vec.sum() == 1:
                idx = np.nonzero(pos_vec)[0][0]
                added = atom_name[idx]

            if removed and added:
                # A + gas1 → B + gas2
                line += f"{rxn0},{added}/{rxn1},{removed}\n"
            elif removed:
                # gas removed only
                line += f"{rxn0}/{rxn1},{removed}\n"
            elif added:
                # gas added only
                line += f"{rxn0},{added}/{rxn1}\n"
            else:
                line += f"{rxn0}/{rxn1}/{comp_change}, needs change!\n"

        return line

def main():
    urg = UniqueRxnGenerator()
    urg.run()

if __name__ == "__main__":
    main()
