#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="oneshot_polyatom"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'

### To choose GPU nodes, turn on the option below...
# export CUDA_VISIBLE_DEVICES= 0

cwd=`pwd`
path_code="${cwd}/select_poscar.py"
for i in `ls -d */`;do
    dst="${i}/oneshot"
    mkdir -p ${dst}
    cp ${cwd}/INCAR_oneshot ${dst}/INCAR
    cp ${i}/rlx/{POTCAR,KPOINTS} ${dst}

    cd ${dst}
    python ${path_code} ../md/OUTCAR

    ##### oneshot #####
    for j in `ls poscars/*`;do
        dst2=`echo ${j} | xargs -n 1 basename`
        mkdir ${dst2}
        cp INCAR KPOINTS POTCAR ${dst2}
        cp ${j} ${dst2}/POSCAR

        cd ${dst2}
        mpiexec.hydra -np $SLURM_NTASKS ${vasp_gam}  >& stdout.x
        cd ..
    done
    ##### oneshot #####

    cd ${cwd}
done
