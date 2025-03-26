#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="mqa_nnp"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

lmp_path="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn"
lmp_input="lammps.in"
path_potential="/data2_1/team_etch/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"

${lmp_path} -v path_potential ${path_potential} -v SEEDS ${RANDOM} -in ${lmp_input}
