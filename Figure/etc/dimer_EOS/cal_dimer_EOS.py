import os
import sys
import pickle
from ase.io import read, write
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class PARAMS_SiO2:
    src_1 = '/data_etch/OUTCAR_trainingset/training_set_OUTCAR/diatomic_EOS'
    src_2 = '/data2/andynn/Etch/01_dft_trset/oneshot/03_gas/01_diatomic'
    path_dict_dft = {
        ('C', 'C'): (f'{src_1}/CC', 60, 148),
        ('C', 'Si'): (f'{src_1}/CSi', 60, 148),
        ('C', 'F'): (f'{src_1}/CF', 60, 148),
        ('C', 'H'): (f'{src_1}/CH', 46, 148),
        ('F', 'F'): (f'{src_1}/hch/FF/eos', -10, 30),
        ('F', 'H'): (f'{src_1}/hch/HF/eos', -10, 30),
        ('H', 'H'): (f'{src_1}/hch/HH/eos', -10, 30),
        ('Si', 'F'): (f'{src_1}/hch/SiF/eos', -10, 30),
        ('Si', 'H'): (f'{src_1}/hch/SiH/eos', -10, 30),
        ('Si', 'Si'): (f'{src_1}/hch/SiSi/eos', -10, 30),
        ('O', 'C'): (f'{src_2}/OC', -10, 30),
        ('O', 'F'): (f'{src_2}/OF', -10, 30),
        ('O', 'O'): (f'{src_2}/OO', -10, 30),
        ('O', 'Si'): (f'{src_2}/OSi', -10, 30),
        ('O', 'H'): (f'{src_2}/OH', -10, 30),
    }

    src_root = "./nnp"
    path_dict_nnp = {
        ('C', 'C'): (f'{src_root}/CC', 60, 148),
        ('C', 'Si'): (f'{src_root}/CSi', 60, 148),
        ('C', 'F'): (f'{src_root}/CF', 60, 148),
        ('C', 'H'): (f'{src_root}/CH', 46, 148),

        ('F', 'F'): (f'{src_root}/FF', -10, 30),
        ('F', 'H'): (f'{src_root}/HF', -10, 30),
        ('H', 'H'): (f'{src_root}/HH', -10, 30),
        ('Si', 'F'): (f'{src_root}/SiF', -10, 30),
        ('Si', 'H'): (f'{src_root}/SiH', -10, 30),
        ('Si', 'Si'): (f'{src_root}/SiSi', -10, 30),

        ('O', 'C'): (f'{src_root}/OC', -10, 30),
        ('O', 'F'): (f'{src_root}/OF', -10, 30),
        ('O', 'O'): (f'{src_root}/OO', -10, 30),
        ('O', 'Si'): (f'{src_root}/OSi', -10, 30),
        ('O', 'H'): (f'{src_root}/OH', -10, 30),
    }

@dataclass
class PARAMS_Si3N4:
    src_dft = "/data_etch/OUTCAR_trainingset/diatomic"
    path_dict_dft = {
        ('C', 'C'): (f'{src_dft}/CC', 60, 148),
        ('C', 'Si'): (f'{src_dft}/CSi', 60, 148),
        ('C', 'F'): (f'{src_dft}/CF', 60, 148),
        ('C', 'H'): (f'{src_dft}/CH', 46, 148),
        ('N', 'C'): (f'{src_dft}/NC', 61, 148),

        ('F', 'F'): (f'{src_dft}/FF', -10, 30),
        ('H', 'F'): (f'{src_dft}/HF', -10, 30),
        ('H', 'H'): (f'{src_dft}/HH', -10, 30),
        ('Si', 'F'): (f'{src_dft}/SiF', -10, 30),
        ('Si', 'H'): (f'{src_dft}/SiH', -10, 30),
        ('Si', 'Si'): (f'{src_dft}/SiSi', -10, 30),

        ('Si', 'N'): (f'{src_dft}/SiN', -10, 30),
        ('N', 'F'): (f'{src_dft}/NF', -10, 30),
        ('N', 'H'): (f'{src_dft}/NH', -10, 30),
        ('N', 'N'): (f'{src_dft}/NN', -10, 30),
    }

    src_nnp = "./nnp"
    path_dict_nnp = {
        ('C', 'C'): (f'{src_nnp}/CC', 60, 148),
        ('C', 'Si'): (f'{src_nnp}/CSi', 60, 148),
        ('C', 'F'): (f'{src_nnp}/CF', 60, 148),
        ('C', 'H'): (f'{src_nnp}/CH', 46, 148),
        ('N', 'C'): (f'{src_nnp}/NC', 61, 148),

        ('F', 'F'): (f'{src_nnp}/FF', -10, 30),
        ('H', 'F'): (f'{src_nnp}/HF', -10, 30),
        ('H', 'H'): (f'{src_nnp}/HH', -10, 30),
        ('Si', 'F'): (f'{src_nnp}/SiF', -10, 30),
        ('Si', 'H'): (f'{src_nnp}/SiH', -10, 30),
        ('Si', 'Si'): (f'{src_nnp}/SiSi', -10, 30),

        ('Si', 'N'): (f'{src_nnp}/SiN', -10, 30),
        ('N', 'F'): (f'{src_nnp}/NF', -10, 30),
        ('N', 'H'): (f'{src_nnp}/NH', -10, 30),
        ('N', 'N'): (f'{src_nnp}/NN', -10, 30),
    }

class DataLoaderNNP:
    def __init__(self, system):
        self.system = system

    def run(self, path_dict):
        path_save = 'data_dimer_NNP.pkl'
        if os.path.exists(path_save):
            print(f'File {path_save} already exists. Loading data...')
            with open(path_save, 'rb') as f:
                data = pickle.load(f)
            print('Data loaded successfully.')
            return data

        result = {}
        for (atom1, atom2), (src, start, end) in path_dict.items():
            print(f'Processing {atom1}-{atom2}...')
            folders = self.gen_folder_list(src, start, end)

            for (idx, folder) in folders:
                energy = self.process_log(f'{folder}/log.lammps')
                if energy is not None:
                    print(f'Processed {folder}: Energy = {energy:.3f} eV')
                else:
                    breakpoint()
                key = (atom1, atom2, idx, folder)
                value = { 'energy': energy, }
                result[key] = value

        with open(path_save, 'wb') as f:
            pickle.dump(result, f)
        print(f'Data saved to {path_save}')
        return result

    def process_log(self, path_log):
        with open(path_log, 'r') as f:
            lines = f.readlines()[::-1]
        for line in lines:
            if line.startswith('free  '):
                energy = float(line.split()[-1])
                return energy

    def gen_folder_list(self, src, start, end):
        if start > 0:
            folders = [(i, f'{src}/{i}') for i in range(start, end + 1)]
        else:
            folders = \
                [(i, f'{src}/_{abs(i)}') for i in range(start, 0)] +\
                [(i, f'{src}/{i}') for i in range(end + 1)]
        return folders

class DataLoaderDFT:
    def __init__(self, system):
        self.system = system

    def run(self, path_dict):
        path_save = 'data_dimer_DFT.pkl'
        if os.path.exists(path_save):
            print(f'File {path_save} already exists. Loading data...')
            with open(path_save, 'rb') as f:
                data = pickle.load(f)
            print('Data loaded successfully.')
            return data

        result = {}
        for (atom1, atom2), (src, start, end) in path_dict.items():
            print(f'Processing {atom1}-{atom2}...')

            folders = self.gen_folder_list(src, start, end)
            for (idx, folder) in folders:
                atoms, energy, bond_length = self.process_OUTCAR(f'{folder}/OUTCAR')
                print(f'Processed {folder}: Energy = {energy:.3f} eV, Bond Length = {bond_length:.3f} Å')
                key = (atom1, atom2, idx, folder)
                value = {
                    'atoms': atoms,
                    'energy': energy,
                    'bond_length': bond_length
                }
                result[key] = value

        with open(path_save, 'wb') as f:
            pickle.dump(result, f)
        print(f'Data saved to {path_save}')
        return result

    def process_OUTCAR(self, path_outcar):
        atoms = read(path_outcar)
        energy = atoms.get_potential_energy()
        bond_length = atoms.get_distance(0, 1, mic=True)
        return atoms, energy, bond_length

    def gen_folder_list(self, src, start, end):
        if self.system == 'SiO2':
            if start > 0:
                folders = [(i, f'{src}/{i}') for i in range(start, end + 1)]
            else:
                if 'hch' in src:
                    folders = \
                        [(i, f'{src}/_{abs(i)}') for i in range(start, 0)] +\
                        [(i, f'{src}/{i}') for i in range(end + 1)]
                else:
                    folders = \
                        [(i, f'{src}/POSCAR__{abs(i)}') for i in range(start, 0)] +\
                        [(i, f'{src}/POSCAR_{i}') for i in range(end + 1)]
            return folders
        elif self.system == 'Si3N4':
            if start > 0:
                folders = [(i, f'{src}/{i}') for i in range(start, end + 1)]
            else:
                folders = \
                    [(i, f'{src}/_{abs(i)}') for i in range(start, 0)] +\
                    [(i, f'{src}/{i}') for i in range(end + 1)]
            return folders

class FileGenerator:
    def __init__(self, system):
        if system == 'SiO2':
            self.atom_types = ['Si', 'O', 'C', 'H', 'F']
        elif system == 'Si3N4':
            self.atom_types = ['Si', 'N', 'C', 'H', 'F']
        else:
            raise ValueError("Invalid system. Use 'SiO2' or 'Si3N4'.")

    def run(self, data):
        for (atom1, atom2, idx, folder), value in data.items():
            atoms = value['atoms']
            if idx < 0:
                idx = f'_{abs(idx)}'
            dst = f'nnp/{atom1}{atom2}/{idx}'
            os.makedirs(dst, exist_ok=True)
            if not os.path.exists(f'{dst}/coo'):
                write(f'{dst}/coo', atoms, format='lammps-data', specorder=self.atom_types)
            print(f'Generated file for {atom1}-{atom2} in {dst}/coo')

class DataProcessor:
    def run(self, data_dft, data_nnp):
        result = {}
        for (atom1, atom2, idx, folder), value in data_dft.items():
            key = (atom1, atom2)
            if key not in result:
                result[key] = {'dft': {}, 'nnp': {}, 'bond_length': {}}
            result[key]['dft'][idx] = value['energy']
            result[key]['bond_length'][idx] = value['bond_length']
        for (atom1, atom2, idx, folder), value in data_nnp.items():
            key = (atom1, atom2)
            if key not in result:
                result[key] = {'dft': {}, 'nnp': {}, 'bond_length': {}}
            result[key]['nnp'][idx] = value['energy']
        return result

class Plotter:
    def __init__(self, system):
        if system == 'SiO2':
            self.atom_types = ['Si', 'O', 'C', 'H', 'F']
        elif system == 'Si3N4':
            self.atom_types = ['Si', 'N', 'C', 'H', 'F']
        else:
            raise ValueError("Invalid system. Use 'SiO2' or 'Si3N4'.")

    def run(self, data, magnify=False):
        fig, ax_dict = self.generate_figure()
        bond_length_mat = {
            'dft': np.zeros((len(self.atom_types), len(self.atom_types))),
            'nnp': np.zeros((len(self.atom_types), len(self.atom_types))),
        }
        for (atom1, atom2), values in data.items():
            ax = ax_dict[(atom1, atom2)]
            x = sorted([i for i in values['dft'].keys()])
            x = np.array(x)
            E_dft = np.array([values['dft'][i] for i in x])
            E_nnp = np.array([values['nnp'].get(i, np.nan) for i in x], dtype=float)
            bond_lengths = np.array([values['bond_length'][i] for i in x])
            E_min_dft = np.nanmin(E_dft)
            E_min_nnp = np.nanmin(E_nnp)

            bl_row, bl_col = self.atom_types.index(atom1), self.atom_types.index(atom2)
            if bl_row > bl_col:
                bl_row, bl_col = bl_col, bl_row
            bond_length_mat['dft'][bl_row, bl_col] = bond_lengths[np.argmin(E_dft)]
            bond_length_mat['nnp'][bl_row, bl_col] = bond_lengths[np.argmin(E_nnp)]
            y_min = min(E_min_dft, E_min_nnp)

            ax.plot(bond_lengths, E_dft, 'o-', label='DFT', color='black', alpha=0.5)
            ax.plot(bond_lengths, E_nnp, 'o-', label='NNP', color='red', alpha=0.5)
            ax.set_title(f'{atom1}-{atom2}')
            ax.set_xlabel('Bond Length (Å)')
            ax.set_ylabel('Energy (eV)')
            ax.legend()

            if magnify:
                text = r'E${}_{min}^{DFT}$' + f' = {E_min_dft:.3f} eV\n' + \
                       r'E${}_{min}^{NNP}$' + f' = {E_min_nnp:.3f} eV\n' + \
                       r'$\Delta$E${}_{min}$' + f' = {E_min_nnp - E_min_dft:.3f} eV'
                ax.text(0.98, 0.02, text, transform=ax.transAxes,
                        ha='right', va='bottom',)
                window = 0.5  # eV
                ax.set_ylim(y_min - window, y_min + window)

        for name, bl_mat in bond_length_mat.items():
            self.save_bondlength_mat(name, bl_mat)

        fig.tight_layout()
        name = 'result'
        fig.savefig(f'{name}.png')

    def generate_figure(self):
        n_row = n_col = len(self.atom_types)
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 3, n_row * 3))
        ax_dict = {}
        for i, atom1 in enumerate(self.atom_types):
            for j, atom2 in enumerate(self.atom_types):
                if i >= j:
                    ax = axes[j, i]
                else:
                    ax = axes[i, j]
                ax_dict[(atom1, atom2)] = ax
        for i, atom1 in enumerate(self.atom_types):
            for j, atom2 in enumerate(self.atom_types):
                if i > j:
                    axes[i, j].axis('off')
        return fig, ax_dict

    def save_bondlength_mat(self, name, bl_mat):
        n_row, n_col = bl_mat.shape
        for row in range(n_row):
            for col in range(n_col):
                if row > col:
                    bl_mat[row, col] = bl_mat[col, row]
        bl_mat *= 1.3
        np.savetxt(f'cutoff_matrix_{name}.npy', bl_mat, fmt='%.3f')

def main():
    if len(sys.argv) != 2:
        print("Usage: python cal_dimer_EOS.py <SiO2|Si3N4>")
        return
    system = sys.argv[1]
    if system == 'SiO2':
        path_dict_dft = PARAMS_SiO2.path_dict_dft
        path_dict_nnp = PARAMS_SiO2.path_dict_nnp
    elif system == 'Si3N4':
        path_dict_dft = PARAMS_Si3N4.path_dict_dft
        path_dict_nnp = PARAMS_Si3N4.path_dict_nnp
    else:
        print("Invalid argument. Use 'SiO2' or 'Si3N4'.")
        return

    dl_dft = DataLoaderDFT(system)
    data_dft = dl_dft.run(path_dict_dft)

    # fg = FileGenerator(system)
    # fg.run(data_dft)

    dl_nnp = DataLoaderNNP(system)
    data_nnp = dl_nnp.run(path_dict_nnp)

    dp = DataProcessor()
    data = dp.run(data_dft, data_nnp)

    pl = Plotter(system)
    pl.run(data, magnify=True)

if __name__ == '__main__':
    main()
