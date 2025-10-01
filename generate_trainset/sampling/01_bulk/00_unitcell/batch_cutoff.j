#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4          # Partition name (skylake)
##
#SBATCH --job-name="cutoff_test"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='vasp_std'
vasp_gam='vasp_gam'
vasp_ncl='vasp_ncl'

### To choose GPU nodes, turn on the option below...
# export CUDA_VISIBLE_DEVICES= 0

src=`pwd`
for i in `ls -d */`;do
    cd ${i}

    mkdir kptest
    mv {2..12} kptest/

    # entest

    for encut in {300..700..50};do
        mkdir ${encut}
        cp POSCAR KPOINTS ${encut}
        cp ${src}/POTCAR ${encut}
        sed "/ENCUT/s/400/${encut}/g" ${src}/INCAR_cutoff > ${encut}/INCAR

        cd ${encut}
        mpiexec.hydra -np $SLURM_NTASKS ${vasp_std} >& stdout.x
        cd ..
    done

    cd ..
done
