#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="relax_gnn"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# lmp_path="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn"
lmp_path="/home/andynn/lammps_sevenn/build/lmp"
path_potential="/data2/andynn/ZBL_modify/codes/pot/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"
# path_potential="/data2/andynn/ZBL_modify/codes/pot/old/old_pot.pt"
# path_potential="/data2/shared_data/pretrained/7net_chgTot/deployed_serial.pt"
lmp_input="lammps.in"

src="/data2/andynn/ZBL_modify/SmallCell/02_NNP_RIE/old_pot/log_20/CF/20"
cwd=$(pwd)
for i in $(ls -d */);do
    cmd="${lmp_path} -in ${lmp_input}"
    cmd="${cmd} -v path_potential ${path_potential}"
    cd ${i}
    eval ${cmd}
    cd ${cwd}
done
