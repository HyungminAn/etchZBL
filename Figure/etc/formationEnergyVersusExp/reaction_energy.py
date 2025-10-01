from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


def energy_filter(src, energy):
    NIONS_Si = 8
    NIONS_C = 12
    NFORMULA_SiO2 = 3
    if 'Si_s' in src:
        energy /= NIONS_Si
    elif 'C_s' in src:
        energy /= NIONS_C
    elif 'SiO2_s' in src:
        energy /= NFORMULA_SiO2
    return energy


def read_energy_DFT(src):
    with open(src, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'free  ' in line:
            energy = float(line.split()[-2])
        if 'NIONS' in line:
            NIONS = int(line.split()[-1])
    energy = energy_filter(src, energy)
    return energy

def read_energy_NNP(src):
    with open(src, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('pe=') and 'eV_' in line:
            energy = float(line.split()[-2])
    energy = energy_filter(src, energy)
    return energy


def make_energy_dict():
    FormationEnergy = namedtuple('FormationEnergy', ['EXP', 'DFT', 'NNP'])
    EXP_E_in_kJ_mol ={
        'SiO2_s': -911.0,
        'C2_g': 820.28,
        'O2_g': 0.0,
        'F2_g': 0.0,

        'CF_g': 243.1700,
        'CF2_g': -193.8900,
        'CF3_g': -465.1500,
        'CF4_g': -927.6300,

        'SiF2_g': -629.1000,
        'SiF4_g': -1608.7600,
        'CO_g': -113.7990,
        'COF2_g': -603.3400,
        'CO2_g': -393.1110,

        'OF2_g': 26.8000,

        'Si_g': 450.34,
        'O_g': 246.844,
        'C_g': 711.3930,
        'F_g': 77.2540,
        }
    kJ_mol_to_eV = 0.010364272301
    src_DFT = "./dft"
    src_NNP = "./nnp"

    data = {}

    for key, E_EXP in EXP_E_in_kJ_mol.items():
        E_DFT = read_energy_DFT(f'{src_DFT}/{key}/OUTCAR')
        E_NNP = read_energy_NNP(f'{src_NNP}/{key}/log.lammps')
        data[key] = FormationEnergy(E_EXP * kJ_mol_to_eV, E_DFT, E_NNP)
        # print(f'{key}: {E_EXP * kJ_mol_to_eV} {E_DFT} {E_NNP}')

    return data


def get_energy(species_list, energy_dict, cal_type):
    total_energy = 0.0
    for (species, coeff) in species_list:
        if cal_type == 'DFT':
            energy = energy_dict[species].DFT
        elif cal_type == 'NNP':
            energy = energy_dict[species].NNP
        elif cal_type == 'EXP':
            energy = energy_dict[species].EXP
        # print(f'{species} {energy:.4f} ({cal_type})')
        total_energy += energy * coeff
    return total_energy


def make_reaction_info(reactants, products):
    line_reactants, line_products = [], []
    for (species, coeff) in reactants:
        line_reactants.append(f'{coeff}{species}')
    for (species, coeff) in products:
        line_products.append(f'{coeff}{species}')
    return '+'.join(line_reactants) + '->' + '+'.join(line_products)


def calculate_reaction_energy():
    # reaction_dict_old = {
    #     'C_g': {
    #         'reactants': [('C_s', 1)],
    #         'products': [('C_g', 1)],
    #         'nions': 1,
    #         },
    #     'F_g': {
    #         'reactants': [('F2_g', 0.5)],
    #         'products': [('F_g', 1)],
    #         'nions': 1,
    #         },
    #     'CF_g': {
    #         'reactants': [('C_s', 1), ('F2_g', 0.5)],
    #         'products': [('CF_g', 1)],
    #         'nions': 2,
    #         },
    #     'CF_g': {
    #         'reactants': [('C_s', 1), ('F2_g', 0.5)],
    #         'products': [('CF_g', 1)],
    #         'nions': 2,
    #         },
    #     'CF2_g': {
    #         'reactants': [('C_s', 1), ('F2_g', 1)],
    #         'products': [('CF2_g', 1)],
    #         'nions': 3,
    #         },
    #     'CF3_g': {
    #         'reactants': [('C_s', 1), ('F2_g', 1.5)],
    #         'products': [('CF3_g', 1)],
    #         'nions': 4,
    #         },
    #     'CF4_g': {
    #         'reactants': [('C_s', 1), ('F2_g', 2)],
    #         'products': [('CF4_g', 1)],
    #         'nions': 5,
    #         },
    #     'SiF2_g': {
    #         'reactants': [('Si_s', 1), ('F2_g', 1)],
    #         'products': [('SiF2_g', 1)],
    #         'nions': 3,
    #         },
    #     'SiF4_g': {
    #         'reactants': [('Si_s', 1), ('F2_g', 2)],
    #         'products': [('SiF4_g', 1)],
    #         'nions': 5,
    #         },
    #     'CO_g': {
    #         'reactants': [('C_s', 1), ('O2_g', 0.5)],
    #         'products': [('CO_g', 1)],
    #         'nions': 2,
    #         },
    #     'COF2_g': {
    #         'reactants': [('C_s', 1), ('O2_g', 0.5), ('F2_g', 1)],
    #         'products': [('COF2_g', 1)],
    #         'nions': 4,
    #         },
    #     'CO2_g': {
    #         'reactants': [('C_s', 1), ('O2_g', 1)],
    #         'products': [('CO2_g', 1)],
    #         'nions': 3,
    #         },
    #     'OF2_g': {
    #         'reactants': [('O2_g', 0.5), ('F2_g', 1)],
    #         'products': [('OF2_g', 1)],
    #         'nions': 3,
    #         },
    #     'Si_g': {
    #         'reactants': [('Si_s', 1)],
    #         'products': [('Si_g', 1)],
    #         'nions': 1,
    #         },
    #     'O_g': {
    #         'reactants': [('O2_g', 0.5)],
    #         'products': [('O_g', 1)],
    #         'nions': 1,
    #         },
    #     }
    reaction_dict = {
        'CF_g': {
            'reactants': [('C2_g', 0.5), ('F2_g', 0.5)],
            'products': [('CF_g', 1)],
            'nions': 2,
            },
        'CF2_g': {
            'reactants': [('C2_g', 0.5), ('F2_g', 1)],
            'products': [('CF2_g', 1)],
            'nions': 3,
            },
        'CF3_g': {
            'reactants': [('C2_g', 0.5), ('F2_g', 1.5)],
            'products': [('CF3_g', 1)],
            'nions': 4,
            },
        'CF4_g': {
            'reactants': [('C2_g', 0.5), ('F2_g', 2)],
            'products': [('CF4_g', 1)],
            'nions': 5,
            },
        'SiF2_g': {
            'reactants': [('SiO2_s', 1), ('F2_g', 1)],
            'products': [('SiF2_g', 1), ('O2_g', 1)],
            'nions': 3,
            },
        # 'SiF4_g': {
        #     'reactants': [('SiO2_s', 1), ('F2_g', 2)],
        #     'products': [('SiF4_g', 1), ('O2_g', 1)],
        #     'nions': 5,
        #     },
        'SiF4_g': {
            'reactants': [('SiO2_s', 1), ('CF2_g', 2)],
            'products': [('SiF4_g', 1), ('CO_g', 2)],
            'nions': 5,
            },
        'CO_g': {
            'reactants': [('C2_g', 0.5), ('O2_g', 0.5)],
            'products': [('CO_g', 1)],
            'nions': 2,
            },
        'COF2_g': {
            'reactants': [('C2_g', 0.5), ('O2_g', 0.5), ('F2_g', 1)],
            'products': [('COF2_g', 1)],
            'nions': 4,
            },
        'CO2_g': {
            'reactants': [('C2_g', 0.5), ('O2_g', 1)],
            'products': [('CO2_g', 1)],
            'nions': 3,
            },
        'OF2_g': {
            'reactants': [('O2_g', 0.5), ('F2_g', 1)],
            'products': [('OF2_g', 1)],
            'nions': 3,
            },

        'C_g': {
            'reactants': [('C2_g', 0.5)],
            'products': [('C_g', 1)],
            'nions': 1,
            },
        'Si_g': {
            'reactants': [('SiO2_s', 1)],
            'products': [('Si_g', 1), ('O2_g', 1)],
            'nions': 1,
            },
        'O_g': {
            'reactants': [('O2_g', 0.5)],
            'products': [('O_g', 1)],
            'nions': 1,
            },
        'F_g': {
            'reactants': [('F2_g', 0.5)],
            'products': [('F_g', 1)],
            'nions': 1,
            },
        }

    energy_dict = make_energy_dict()
    reaction_types = ['EXP', 'DFT', 'NNP']

    result = {key: {} for key in reaction_dict}
    result_per_atom = {key: {} for key in reaction_dict}
    for species, reaction_info in reaction_dict.items():
        reactants = reaction_info['reactants']
        products = reaction_info['products']
        nions = reaction_info['nions']
        line = f'{species} ' + make_reaction_info(reactants, products)

        for reaction_type in reaction_types:
            E_reactants = get_energy(reactants, energy_dict, reaction_type)
            E_products = get_energy(products, energy_dict, reaction_type)
            E_reaction = E_products - E_reactants
            line += f" {E_reaction:.4f} ({reaction_type})"

            result[species][reaction_type] = E_reaction
            result_per_atom[species][reaction_type] = E_reaction / nions
        print(line)

    return result, result_per_atom


def plot(result, per_atom=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    key = sorted(result.keys())
    data_EXP = np.array([result[k]['EXP'] for k in key])
    data_DFT = np.array([result[k]['DFT'] for k in key])
    data_NNP = np.array([result[k]['NNP'] for k in key])
    labels = [(key[i], data_EXP[i], data_DFT[i], data_NNP[i]) for i in range(len(key))]

    xy_min = min(data_EXP.min(), data_DFT.min(), data_NNP.min()) * 1.1
    xy_max = max(data_EXP.max(), data_DFT.max(), data_NNP.max()) * 1.1

    for E_EXP, E_DFT, E_NNP in zip(data_EXP, data_DFT, data_NNP):
        ax.plot([E_EXP, E_EXP], [E_DFT, E_NNP], color='grey', alpha=0.5)
    ax.scatter(data_EXP, data_DFT, label='DFT', color='black', alpha=0.5)
    ax.scatter(data_EXP, data_NNP, label='NNP', color='red', alpha=0.5)

    ax.axline((0, 0), slope=1, color='black', linestyle='--')
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.set_title("Reaction Energy (EXP/DFT/NNP)")

    if per_atom:
        ax.set_xlabel('EXP (eV/atom)')
        ax.set_ylabel('DFT/NNP (eV/atom)')
    else:
        ax.set_xlabel('EXP (eV)')
        ax.set_ylabel('DFT/NNP (eV)')

    texts = []
    x, y = data_EXP, data_DFT
    for i, (key, E_EXP, E_DFT, E_NNP) in enumerate(labels):
        txt = f'{key} ({E_EXP:.2f}/{E_DFT:.2f}/{E_NNP:.2f})'
        txt_position = (x[i], y[i])
        texts.append(ax.text(txt_position[0], txt_position[1], txt, fontsize=10))

    adjust_text_props = {
        'arrowprops': dict(arrowstyle='->', color='red', lw=0.5),
        'expand': (1.0, 2.0),
    }
    adjust_text(texts, x, y, **adjust_text_props)

    fig.tight_layout()
    if per_atom:
        fig.savefig('reaction_energy_per_atom.png')
    else:
        fig.savefig('reaction_energy.png')


def main():
    result, result_per_atom = calculate_reaction_energy()
    plot(result, per_atom=False)
    plot(result_per_atom, per_atom=True)


if __name__ == '__main__':
    main()
