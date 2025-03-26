#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24         # Cores per node
#SBATCH --partition=loki3          # Partition name (skylake)
##
#SBATCH --job-name="rlx_nnp"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

cwd=$(pwd)
lmp_path="/home/andynn/lammps/build_etch_d2_loki34/lmp"

for i in $(ls -d */);do
    cd ${i}
    if [[ ${name} == *"_s" ]];then
        lmp_input="${cwd}/lammps_s.in"
    else
        lmp_input="${cwd}/lammps.in"
    fi
    ${lmp_path} -in ${lmp_input}
    cd ${cwd}
done
