#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="nnp_mol_relax"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

lmp_path="/home/andynn/lammps_sevenn/build/lmp"
# lmp_path="/data2_1/team_etch/LAMMPS/lammps-2Aug2023/build/lmp"
path_potential="/data2_1/team_etch/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"
# path_potential="/data2/shared_data/pretrained/7net_chgTot/deployed_serial.pt"

cwd=$(pwd)
lmp_input="${cwd}/lammps.in"
for i in CF CF2 CF3 CHF CHF2 CH2F;do
    cd ${i}
    ${lmp_path} -v path_potential ${path_potential} -v SEEDS ${RANDOM} -in ${lmp_input}
    cd ..
done
