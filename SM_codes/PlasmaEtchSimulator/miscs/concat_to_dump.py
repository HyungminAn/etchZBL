#!/bin/python

# This script open the divserse files to dump.lammps to calculate it using rerun !!
# The filename or index ins saved to other files which cannot saved to dump.lammps 
import os
import mendeleev
import ase
import ase.io
import glob
from ase.data import chemical_symbols
def _lmp_rerun(elmlist, fout, pottype = 'e3gnn',potloc = '/home/gasplant63/pot_7net_chgtot_l3i3_zbl.pt', alpha=0.2):
    mass = ''
    pair_coeff = ''
    ndict = {}
    for it, elm in enumerate(elmlist):
        mass += f'mass  {it+1} {mendeleev.element(elm).atomic_weight} \n'

        for init, inelm in enumerate(elmlist[it:]):
            if elm not in ndict:    
                ndict[elm] = mendeleev.element(elm).atomic_number 
            if inelm not in ndict:    
                ndict[inelm] = mendeleev.element(inelm).atomic_number 

            pair_coeff += f'pair_coeff {it+1} {it+init+1} {ndict[elm]} {ndict[inelm]}\n'

    if pottype == 'e3gnn':
        potin = f'pair_style e3gnn'
        pair_coeff = ''
    elif pottype == 'e3gnn/zbl':
        potin = f' pair_style e3gnn/zbl 0.069 0.139  {alpha}'


    lines = f'''units           metal
newton          on
dimension       3
boundary        p p p
atom_style 	atomic      
box tilt large

read_data base.coo
{mass}

 {potin}
 pair_coeff * *  {potloc}  {' '.join(elmlist)}

{pair_coeff}

thermo_style    custom   step pe 
thermo          1
dump            my_dump     all custom 1 ${{dumpout}} &
                            id type element xu yu zu fx fy fz
dump_modify     my_dump     element {' '.join(elmlist)}

fix termp all print 1 "$(step) $(pe)" file ${{thermo}} screen no

rerun ${{dumpin}} dump x y z box yes 
'''
    with open(fout, 'w') as f:
        f.write(lines)
        print(f"{fout} Written")

def _create_dump_for_rerun(elmlist : list, filelist : list, dumpname : str, indexname :str) -> None:
    """This fucntion create lammps dump file which calculated to rerun for specific potential

    Args:
        elmlist (list): the element list to convert to dump file
        filelist (list): the file list to convert to dump file
        dumpname (str): the name of dump which saved
        indexname (str): the name of index file which saved 
    """
    # Create convert dictionnay from symbol H He Li Be .. -> elmlist[0], elmlist[1], elmlist[2] ...

    sym_conv = { chemical_symbols[it+1]: elmlist[it] for it in range(len(elmlist))}

    dumpfile = open(dumpname, 'w')
    indexfile = open(indexname, 'w')
    for it, fname in enumerate(filelist):
        if 'POSCAR' in fname or 'CONTCAR' in fname:
            atoms = ase.io.read(fname, format='vasp')
        else:
            atoms = ase.io.read(fname, format='lammps-data')

        dumpfile.write(f'ITEM: TIMESTEP\n{it}\n')
        dumpfile.write(f'ITEM: NUMBER OF ATOMS\n{len(atoms)}\n')
        dumpfile.write(f'ITEM: BOX BOUNDS pp pp pp\n')
        for i in range(3):
            dumpfile.write(f'{0.0} {atoms.cell[i,i]}\n')
        dumpfile.write(f'ITEM: ATOMS id element xu yu zu fx fy fz\n')
        for i, atom in enumerate(atoms):
            dumpfile.write(f'{i+1} {sym_conv[atom.symbol]} {atom.position[0]} {atom.position[1]} {atom.position[2]} 0 0 0\n')
        indexfile.write(f'{fname}\n')

    print(f"Dump file {dumpname} and index file {indexname} created")

def _split_to_extxyz(outdump : str, thermofile : str, indexfile : str, savedir : str, index_parser = None) -> None:
    """This function read outdump files which calculated using LAMMPS at once and split to each files 
    ising index files to savedir

    Args:
        outdump (str): the dump file which saved all the structure
        thermofile (str): the thermo file which saved the energy 
        indexfile (str): the index file which saved the filename
        savedir (str): the directory to save the files
        index_parser ([type], optional): the function to parse the index file. Defaults to None.
    """

    os.makedirs(savedir, exist_ok=True)
    atoms = ase.io.read(outdump, format='lammps-dump-text', index='::' )
    with open(indexfile, 'r') as f:
        filelist = f.readlines()
    with open(thermofile, 'r') as f:
        thermolist = f.readlines()[1:]

    assert len(atoms) == len(filelist), "The number of atoms and filelist is not matched"
    assert len(atoms) == len(thermolist), "The number of atoms and thermolist is not matched"
    for at, fpath, therm in zip(atoms, filelist, thermolist):
        fpath = fpath.strip()
        if index_parser is not None:
            fdir, fname  = index_parser(savedir, fpath)
        else:
            fdir  = savedir
            fname = fpath.split('/')[-2]+'_'+fpath.split('/')[-1]+'.extxyz'

        forces = at.get_forces()
        os.makedirs(fdir, exist_ok=True)
        at.calc = None
        at.info['energy'] = float(therm.split()[1])
        at.arrays['forces'] = forces
        at.write(f'{fdir}/{fname}', format='extxyz')    


if __name__  =='__main__':
    elmlist = ['Si', 'N', 'O', 'C', 'H', 'F']
    gendump     = False
    lmpin       = True
    lmprun      = True
    savedump    = False
    if gendump:
        filelist = sorted(glob.glob('/data2/gasplant63/etch_gnn/7_SiNOCHF/9_lammps_mod/6_test_lmp_case/threebody_fit/coo_*/coo_*'))
        #Sort filelist with numerical and alphabetic order
        runlist = filelist  # This is to test the rerun for the first 10 files   
        _create_dump_for_rerun(elmlist, runlist, 'dump_rerun_in.lammps', 'index.txt')
        print("Generate dump files")

    data    = 'data/'
    dumpin  = f'data/dump_rerun_in.lammps'
    dumpout = f'data/dump_rerun_base.lammps'
    thermo  = f'data/thermo.dat'
    log     = f'data/log_base.lammps'
    extxyzdir = 'base_extxyz'

    if lmpin:
        _lmp_rerun(elmlist, '/data2/gasplant63/etch_gnn/7_SiNOCHF/9_lammps_mod/6_test_lmp_case/threebody_fit/alpha_fit/rerun.lmp' ,pottype='e3gnn')  
        print("Generate inputs")
    if lmprun:
        lmploc ='/home/gasplant63/lmp_e3gnn_zbl'
        cmd = f'{lmploc} -in /data2/gasplant63/etch_gnn/7_SiNOCHF/9_lammps_mod/6_test_lmp_case/threebody_fit/alpha_fit/rerun.lmp \
            -v dumpin {dumpin} -v dumpout {dumpout} -v thermo {thermo} -l {log} >& stdout.x'
        cnum = os.system(cmd)
        print("Run LAMMPS")
    if savedump:
        _split_to_extxyz(dumpout, thermo, 'index.txt', extxyzdir)
    ## Grid test for alpha [0.2 0.5 1.0 1.5 2.0]
    lmploc ='/home/gasplant63/lmp_e3gnn_zbl'

    for alpha in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:    
        dumpout     = f'data/dump_rerun_a{alpha}.lammps'
        thermo      = f'data/thermo_a{alpha}.dat'
        extxyzdir   = f'a{alpha}_extxyz'
        log         = f'data/log_a{alpha}.lammps'
        _lmp_rerun(elmlist, '/data2/gasplant63/etch_gnn/7_SiNOCHF/9_lammps_mod/6_test_lmp_case/threebody_fit/alpha_fit/rerun.lmp' ,pottype='e3gnn/zbl', alpha=alpha)  
        print(f"Generate inputs : alpha {alpha}")
        cmd = f'{lmploc} -in /data2/gasplant63/etch_gnn/7_SiNOCHF/9_lammps_mod/6_test_lmp_case/threebody_fit/alpha_fit/rerun.lmp \
            -v dumpin {dumpin} -v dumpout {dumpout} -v thermo {thermo} -l {log} >& stdout.x'
        cnum = os.system(cmd)
        print(f"Run LAMMPS : alpha {alpha}")
        _split_to_extxyz(dumpout, thermo, 'index.txt', extxyzdir)
        print(f"Save extxyz : alpha {alpha}")


