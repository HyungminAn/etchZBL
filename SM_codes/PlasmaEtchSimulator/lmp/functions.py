import re
from PlasmaEtchSimulator.params import MoleculeInfo


def _mole_to_ndict(name : str):
    """Convert the molecule name to dictionary of atom name and number
    Example : CF2 => {'C':1, 'F':2}, CH2F => {'C':1, 'H':2, 'F':1} ...
    """
    re.findall(r'[A-Z][a-z]*\d*', name)
    ndict = {}
    for atom in re.findall(r'[A-Z][a-z]*\d*', name):
        if atom[-1].isdigit() and not atom[0] in ndict:
            ndict[atom[:-1]] = int(atom[-1])
        elif atom[-1].isdigit() and atom[0] in ndict:
            ndict[atom[:-1]] += int(atom[-1])
        elif not atom[0] in ndict:
            ndict[atom] = 1
        else:
            ndict[atom] += 1
    return ndict


def ion_writer(fname : str, ion : str, elmlist : list):
    pos_dict = MoleculeInfo.pos_dict
    assert ion in pos_dict, f"ion {ion} is not supported"

    f = open(fname, 'w')
    ndict = _mole_to_ndict(ion)
    tnum = sum(ndict.values())

    posline = ''
    typeline = ''
    count = 1
    for value in pos_dict[ion]:
        elm, pos = value
        posline += f'{count} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n'
        typeline += f'{count} {elmlist.index(elm)+1}\n'
        count += 1

    lines = f'''# Molecule {ion}
{tnum} atoms

Coords

{posline}

Types

{typeline}'''
    f.write(lines)
    f.close()
    print(f"File {fname} written")
