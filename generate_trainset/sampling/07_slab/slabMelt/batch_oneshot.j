#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4          # Partition name (skylake)
##
#SBATCH --job-name="slab_melt_oneshot"
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'

for j in anneal melt liquid;do
    cd ${j}/oneshot
    python select_poscar.py ../OUTCAR

    for i in `ls poscars/*`;do
        dst=`echo ${i} | xargs -n 1 basename`
        mkdir ${dst}
        cp INCAR KPOINTS POTCAR ${dst}
        cp ${i} ${dst}/POSCAR

        cd ${dst}
        mpiexec.hydra -np $SLURM_NTASKS ${vasp_gam}  >& stdout.x
        cd ..
    done


    cd ../../
done
