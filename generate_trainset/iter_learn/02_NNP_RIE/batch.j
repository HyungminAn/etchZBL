#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="JOBTITLE"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

lmp_path="/home/andynn/lammps_sevenn/build/lmp"
path_potential="/data2_1/team_etch/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"
lmp_input="lammps_iterative_SiOCHF.in"

${lmp_path} -v SEEDS ${RANDOM} -v path_potential ${path_potential} -in ${lmp_input}
