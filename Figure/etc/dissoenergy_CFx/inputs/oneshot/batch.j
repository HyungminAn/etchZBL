#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="nnp_oneshot"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

lmp_path="/home/andynn/lammps/build_etch_d2_loki34/lmp"
cwd=$(pwd)
lmp_input="/data2_1/andynn/Etch/07_CFx_dissociation/inputs/oneshot/lammps.in"

for i in $(ls -d */);do
    cd ${i}
    mpirun -np $SLURM_NTASKS ${lmp_path} -in ${lmp_input}
    cd ${cwd}
done
