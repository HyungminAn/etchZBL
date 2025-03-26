#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2         # Partition name (skylake)
##
#SBATCH --job-name="JOBTITLE"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

lmp_path="/home/andynn/lammps_sevenn/build/lmp"
start_path=$PWD
code_path="/data2/andynn/ZBL_modify/codes/SmallCell/05_reactionE/codes"
pt_path="/data2_1/team_etch/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"

path_rlx=${code_path}/lammps_rlx_in/lammps_rlx_gnn.in
path_rlx_gas=${code_path}/lammps_rlx_in/lammps_rlx_gas_gnn.in
path_rlx_gas_zbl=${code_path}/lammps_rlx_in/lammps_rlx_gas_gnn_zbl.in

relax() {
    local lmp_in=$1
    local image_list=$2

    for to_go in $(cat ${image_list});do
        echo ${to_go} >> currently_gnn
        mkdir ${to_go}/gnn/
        cd ${to_go}/gnn/

        ln -s ${lmp_in} lammps_rlx.in
        cp ../coo ./
        ${lmp_path} -in lammps_rlx.in -var pot_path ${pt_path} > lammps_gnn.out
        grep 'free  ' lammps_gnn.out > e

        cd ${start_path}
    done
}

################################################################################
#                                  bulk relax                                  #
################################################################################
ls -d post_process_bulk_gas/*/*_*/ -d > image_list
relax ${path_rlx} image_list

################################################################################
#                                  gas relax                                   #
################################################################################
ls -d post_process_bulk_gas/gas/*/*_*/ > gas_list
relax ${path_rlx_gas} gas_list

################################################################################
#                                 gas trivial                                  #
################################################################################
ls -d post_process_bulk_gas/gas/trivial/*/ > gas_list
relax ${path_rlx_gas_zbl} gas_list
