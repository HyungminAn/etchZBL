#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4          # Partition name (skylake)
##
#SBATCH --job-name="oneshot_ref"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

cwd=$(pwd)
# pot_final
path_potential="/data2/andynn/Etch/03_train/02_LargeSize_Iter3/05_train_401to600/potential_saved_bestmodel"

lmp_path="/home/andynn/lammps/build_etch_d2_loki34/lmp"
lmp_input="${cwd}/lammps.in"

pot_list="pot_0 pot_1 pot_2"
ion_list="CF CF3 CH2F CHF2"
energy_list="10 30"
incidence_list=$(seq 1 1 50)
idx_list=$(seq 0 1 4)

for pot in ${pot_list};do
for ion in ${ion_list};do
for energy in ${energy_list};do
for incidence in ${incidence_list};do
for idx in ${idx_list};do
    src="${cwd}/${pot}/${ion}/${energy}/${incidence}/${idx}"
    cd ${src}
    CMD_LAMMPS="mpirun -np $SLURM_NTASKS ${lmp_path} -in ${lmp_input}"
    CMD_LAMMPS="${CMD_LAMMPS} -v path_potential ${path_potential}"
    ${CMD_LAMMPS}
    cd ${cwd}
done
done
done
done
done
