import os
from itertools import product
import pickle

import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write
from params import PARAMS


class CalculateSetGenerator:
    def run(self):
        result = self.extract_images()
        self.write_image_lammps(result)
        result = self.save_calculation_results()
        # result = self.extract_images()
        # self.write_image_vasp(result)

    def extract_images(self):
        pot_type = PARAMS.pot_type
        ion_list = PARAMS.ion_list
        energy_list = PARAMS.energy_list
        incidences = PARAMS.incidences
        result = {}
        log_time = {}

        for pot, ion, energy, inc in product(pot_type, ion_list, energy_list, incidences):
            src = f"{PARAMS.src}/{pot}/02_NNP_RIE/{ion}/{energy}"
            dump = read(f"{src}/dump_{inc}.lammps", **PARAMS.LAMMPS_READ_OPTS)
            thermo = np.loadtxt(f"{src}/thermo_{inc}.dat", skiprows=1)
            thermo = np.unique(thermo, axis=0)
            time = thermo[:, 1]
            start, end = time[0], time[-1]
            checkpoints = np.arange(start, end + PARAMS.sample_interval, PARAMS.sample_interval)
            mask = np.searchsorted(time, checkpoints, side='left')[:PARAMS.n_max_sample]
            images = [image for idx, image in enumerate(dump) if idx in mask]
            for (idx, image), t in zip(enumerate(images), time[mask]):
                key = (pot, ion, energy, inc, idx)
                result[key] = image
                log_time[key] = t
                print(f"Processed {key}, time: {t}")

        with open('log_time.txt', 'w') as f:
            f.write('# Pot Ion Energy Incidence Index Time\n')
            for key, time in log_time.items():
                pot, ion, energy, inc, idx = key
                f.write(f"{pot} {ion} {energy} {inc} {idx} {time:.2f}\n")
        return result

    def write_image_lammps(self, result):
        for (pot, ion, energy, inc, idx), image in result.items():
            dst = f"{pot}/{ion}/{energy}/{inc}/{idx}"
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
            write(f"{dst}/input.data", image, **PARAMS.LAMMPS_SAVE_OPTS)
            print(f"Written {dst}/input.data")

    def write_image_vasp(self, result):
        for (pot, ion, energy, inc, idx), image in result.items():
            dst = f"{pot}/{ion}/{energy}/{inc}/{idx}"
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
            write(f"{dst}/POSCAR", image, **PARAMS.VASP_SAVE_OPTS)
            print(f"Written {dst}/POSCAR")

    def save_calculation_results(self):
        pot_type = PARAMS.pot_type
        ion_list = PARAMS.ion_list
        energy_list = PARAMS.energy_list
        incidences = PARAMS.incidences
        result = {}

        for pot, ion, energy, inc in product(pot_type, ion_list, energy_list, incidences):
            src = f"{PARAMS.src}/{pot}/02_NNP_RIE/{ion}/{energy}"
            dump = read(f"{src}/dump_{inc}.lammps", **PARAMS.LAMMPS_READ_OPTS)
            thermo = np.loadtxt(f"{src}/thermo_{inc}.dat", skiprows=1)
            thermo = np.unique(thermo, axis=0)
            time = thermo[:, 1]
            start, end = time[0], time[-1]
            checkpoints = np.arange(start, end + PARAMS.sample_interval, PARAMS.sample_interval)
            mask = np.searchsorted(time, checkpoints, side='left')[:PARAMS.n_max_sample]
            pot_e = thermo[:, 3]
            energies = pot_e[mask]
            forces = [image.get_forces(apply_constraint=False) for idx, image in enumerate(dump) if idx in mask]
            for idx, (pot_e, force) in enumerate(zip(energies, forces)):
                key = (pot, ion, energy, inc, idx)
                result[key] = (pot_e, force)
                print(f"Processed {key}, pot_e: {pot_e}")

        with open('EF_original.pkl', 'wb') as f:
            pickle.dump(result, f)
            print(f"Saved calculation results to 'calculation_results.pickle'")
        return result

def main():
    csg = CalculateSetGenerator()
    csg.run()

if __name__ == "__main__":
    main()
