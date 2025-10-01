#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2        # Partition name (skylake)
##
#SBATCH --job-name="bp_test"
#SBATCH --time=00-01:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

#export CUDA_VISIBLE_DEVICES=0

lmp_path="/home/andynn/lammps/build_etch_d2/lmp"
start_path=$PWD
code_path="/data2/andynn/ZBL_modify/codes/SmallCell/05_reactionE/codes"
pt_path="/data2/andynn/Etch/05_EtchingMD/ver2/potential_saved_bestmodel"

path_rlx=${code_path}/lammps_rlx_in/lammps_rlx_bpnn.in
path_rlx_gas=${code_path}/lammps_rlx_in/lammps_rlx_gas_bpnn.in

relax() {
    local lmp_in=$1
    local image_list=$2

    for to_go in $(cat image_list);do
        echo $to_go >> currently_bpnn
        mkdir -p $to_go/bpnn/
        cd $to_go/bpnn/

        rm lammps_rlx.in
        ln -s ${lmp_in} lammps_rlx.in
        cp ../coo ./
        mpirun -np $SLURM_NTASKS  $lmp_path -in lammps_rlx.in -var path_potential ${pt_path} > lammps_bpnn.out
        grep 'free  ' lammps_bpnn.out > e

        cd $start_path
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
relax ${path_rlx_gas} gas_list
