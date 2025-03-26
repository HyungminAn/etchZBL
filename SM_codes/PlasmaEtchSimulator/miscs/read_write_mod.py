import numpy as np
import ase,os,sys
import ase.io
import ase.build

total_iter=50

coll_st     = 0
coll_end    = 2000
coll_step   = 500
rlx_step    = 1000


def z_pos(snap):
    maxpos = snap.get_positions()
    out = np.max(maxpos[:,2])>=30.0

    return out

cut_function = z_pos



conv_dict = dict()
for elm, num in zip("H He Li Be B".split(), [14, 7, 6, 1, 9]):
    conv_dict[elm] = num



for i in range(1,total_iter+1):
    print(f"currently incidence {i}")
    dir_idx=0
    os.makedirs(f"structures/{i}",exist_ok=True)
    atom        =ase.io.read(f"./dump_{i}.lammps",format='lammps-dump-text',index="1:")

    ## COLL
    snap_list   =list(range(coll_st, coll_end, coll_step))
    snap_list   +=list(range(coll_end,len(atom),rlx_step))

    snap_list=np.array(snap_list)
    for snap_idx in snap_list:
        snapshot = atom[snap_idx]
        chem_syms=snapshot.get_chemical_symbols()
        for j in range(len(snapshot)):
            snapshot.numbers[j] = conv_dict[chem_syms[j]]

        ## To Check not reactive cases
        
        if cut_function(snapshot):
            continue
        else:
            os.makedirs(f"structures/{i}/{dir_idx}",exist_ok=True)
            ase.io.write(f"structures/{i}/{dir_idx}/POSCAR",snapshot,format='vasp',sort=True,label=f"incidence:{i} / step {snap_idx+1}")
            dir_idx+=1
    print(f"structures/{i}/ written with {len(snap_list)} POSCARs")

        

