import os
from itertools import product
import pickle

from ase.io import read

from params import PARAMS

class OneshotResultProcessor:
    def run(self):
        path_save = 'EF_oneshot.pkl'
        if os.path.exists(path_save):
            print(f"Loaded existing results from {path_save}...")
            with open(path_save, 'rb') as f:
                result = pickle.load(f)
            return result

        missing_data = []

        pot_type = PARAMS.pot_type
        ion_list = PARAMS.ion_list
        energy_list = PARAMS.energy_list
        incidences = PARAMS.incidences
        result = {}

        for pot, ion, ion_energy, inc, idx in product(pot_type, ion_list, energy_list, incidences, range(PARAMS.n_max_sample)):
            src = f'{pot}/{ion}/{ion_energy}/{inc}/{idx}'
            key = (pot, ion, ion_energy, inc, idx)
            energy = self.read_energy(src)
            force = self.read_force(src)
            if energy is None or force is None:
                print(f"Missing data for {key}, skipping...")
                missing_data.append(key)
                continue

            result[key] = (energy, force)
            print(f"processed {key}, energy: {energy:.2f}")

        with open('EF_oneshot.pkl', 'wb') as f:
            pickle.dump(result, f)

        with open('missing_data.txt', 'w') as f:
            for data in missing_data:
                pot, ion, ion_energy, inc, idx = data
                f.write(f"{pot}/{ion}/{ion_energy}/{inc}/{idx}\n")
        return result

    def read_energy(self, src):
        path_log = f'{src}/log.lammps'
        if not os.path.exists(path_log):
            print(f"Log file {path_log} does not exist.")
            return None
        with open(path_log, 'r') as f:
            lines = f.readlines()
        for line in lines[::-1]:
            if line.startswith('free  '):
                energy = float(line.split()[-1])
        return energy

    def read_force(self, src):
        if not os.path.exists(f'{src}/dump.lammps'):
            print(f"Dump file {src}/dump.lammps does not exist.")
            return None
        path_dump = f'{src}/dump.lammps'
        dump = read(path_dump, format='lammps-dump-text')
        force = dump.get_forces(apply_constraint=False)
        return force

def main():
    orp = OneshotResultProcessor()
    orp.run()

if __name__ == "__main__":
    main()
