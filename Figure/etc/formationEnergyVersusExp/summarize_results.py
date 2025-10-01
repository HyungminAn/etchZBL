import os

class OutcarProcessor:
    def run(self, path_outcar):
        is_converged = self.check_converged_OUTCAR(path_outcar)
        energies = self.get_energy_from_OUTCAR(path_outcar)
        nions = self.get_nions(path_outcar)
        result = {
            "is_converged": is_converged,
            "energies": energies,
            "nions": nions,
            "energies_per_atom": [energy / nions for energy in energies] if nions else []
        }
        return result

    def check_converged_OUTCAR(self, path_outcar):
        with open(path_outcar, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if "reached required accuracy" in line:
                return True
        return False

    def get_energy_from_OUTCAR(self, path_outcar):
        with open(path_outcar, 'r') as file:
            lines = file.readlines()

        result = []
        for line in lines:
            if "free  energy   TOTEN" in line:
                parts = line.split()
                result.append(float(parts[-2]))
        return result

    def get_nions(self, path_outcar):
        with open(path_outcar, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if "NIONS" in line:
                parts = line.split()
                return int(parts[-1])
        return None

def main():
    src_dft = "dft"
    src_nnp = "nnp"

    species_list = [i for i in os.listdir(src_dft)]
    op = OutcarProcessor()
    for folder in species_list:
        path_outcar = os.path.join(src_dft, folder, "OUTCAR")
        result = op.run(path_outcar)
        if folder.endswith('_s'):
            print(f"{folder} {result['energies_per_atom'][-1]} (eV/atom)")
        else:
            print(f"{folder} {result['energies'][-1]} (eV)")

if __name__ == "__main__":
    main()
