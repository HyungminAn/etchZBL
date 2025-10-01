from ase.io import read, write
from tempfile import NamedTemporaryFile as NTF
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


def compress_outcar(filename, res):
    """
    *** From SIMPLE-NN code ***

    Compress VASP OUTCAR file for fast file-reading in ASE.
    Compressed file (tmp_comp_OUTCAR)
        is temporarily created in the current directory.

    :param str filename: filename of OUTCAR

    supported properties:

    - atom types
    - lattice vector(cell)
    - free energy
    - force
    - stress
    """

    with open(filename, 'r') as fil:
        minus_tag = 0
        line_tag = 0
        ions_key = 0
        for line in fil:
            if 'POTCAR:' in line:
                res.write(line)
            if 'POSCAR:' in line:
                res.write(line)
            elif 'ions per type' in line and ions_key == 0:
                res.write(line)
                ions_key = 1
            elif 'direct lattice vectors' in line:
                res.write(line)
                minus_tag = 3
            elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
                res.write(line)
                minus_tag = 4
            elif 'POSITION          ' in line:
                res.write(line)
                line_tag = 3
            elif 'FORCE on cell =-STRESS' in line:
                res.write(line)
                minus_tag = 15
            elif 'Iteration' in line:
                res.write(line)
            elif minus_tag > 0:
                res.write(line)
                minus_tag -= 1
            elif line_tag > 0:
                res.write(line)
                if '-------------------' in line:
                    line_tag -= 1


def read_dft_force(path_dft):
    with NTF(
            mode='w+', encoding='utf-8', delete=True, prefix='OUTCAR_'
            ) as tmp_outcar:
        compress_outcar(path_dft, tmp_outcar)
        tmp_outcar.flush()
        tmp_outcar.seek(0)
        path_dft_compressed = tmp_outcar.name
        outcar_dft = read(path_dft_compressed)
        f_dft = outcar_dft.get_forces(apply_constraint=False)
        symbols = outcar_dft.get_chemical_symbols()
    print(f'{path_dft} complete')
    return f_dft, symbols


def make_new_dump(path_dft, path_nnp):
    force_dft, symbols = read_dft_force(path_dft)
    poscar_nnp = read(path_nnp)
    poscar_nnp.set_chemical_symbols(symbols)
    poscar_nnp.set_array('disp', force_dft)
    try:
        assert len(poscar_nnp) == len(symbols)
    except AssertionError:
        breakpoint()
    return poscar_nnp


def write_total_dump(path_dft_list, path_nnp_list):
    with ThreadPoolExecutor() as executor:
        futures = []

        for path_dft, path_nnp in zip(path_dft_list, path_nnp_list):
            futures.append(
                executor.submit(
                    make_new_dump,
                    path_dft,
                    path_nnp,
                )
            )

        total_dump = [future.result() for future in as_completed(futures)]

    write('total_dump_with_force.xyz', total_dump, format='extxyz')


def main():
    path_lammps_outs = 'lammps_outs'
    path_dft_oneshot = '../../03_dft_oneshot'

    ion_type_list = [
        'CF_10',
        'CF_30',
        'CF3_10',
        'CF3_30',
        'CH2F_10',
        'CH2F_30',
        'CHF2_10',
        'CHF2_30',
    ]

    path_dft_list = [
        f'{path_dft_oneshot}/{ion_type}/POSCAR_{i}_{j}/OUTCAR'
        for ion_type in ion_type_list
        for i in range(1, 51)
        for j in range(5)
    ]

    path_nnp_list = [
        f'{path_lammps_outs}/dump_{i}.lammpstrj'
        for i in range(1, 2001)
    ]

    write_total_dump(path_dft_list[:250], path_nnp_list[:250])


main()
