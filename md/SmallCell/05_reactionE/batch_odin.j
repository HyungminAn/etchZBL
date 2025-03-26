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

function check_conda_env() {
    local conda_env=$1

    # Check if 'conda' command is available
    if ! command -v conda &> /dev/null; then
        echo "Error: 'conda' command not found. Ensure that Anaconda or Miniconda is installed."
        exit 1
    fi

    # Initialize conda (adjust the path if necessary)
    source "$(conda info --base)/etc/profile.d/conda.sh"

    # Activate 'orbNet' conda environment
    conda activate ${conda_env}

    # Verify that the environment was activated
    if [[ "$CONDA_DEFAULT_ENV" != "${conda_env}" ]]; then
        echo "Error: Failed to activate <<<${conda_env}>>> conda environment."
        exit 1
    fi
}

check_conda_env "graph-tool"

cal_type="my_cal_type"
code_root="/data2/andynn/ZBL_modify/codes/SmallCell/05_reactionE/codes/"
path_dat="/data2/andynn/ZBL_modify/SmallCell/02_NNP_RIE/new/${cal_type}"

#################################################################################
##                                   Part. 1                                    #
#################################################################################
#python3 $code_root/graph_similarity_save_pickle.py ${path_dat}
#python3 $code_root/save_snapshots_coo.py ${path_dat}

#################################################################################
##                                   Part. 2                                    #
#################################################################################
# lmp_path="/home/andynn/lammps_sevenn/build/lmp"
# lmp_in="lammps_rlx_gnn.in"
# pt_path="/data2_1/team_etch/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"
# ls -d incidences/*/* > mat_list
# for i in $(cat mat_list);do
#     echo $i >> currently

#     cd $i
#     ln -s $code_root/lammps_rlx_in/${lmp_in} ${lmp_in}
#     $lmp_path -var pot_path $pt_path -in ${lmp_in} > lammps.out
#     cd -
# done

#################################################################################
##                                   Part. 3                                    #
#################################################################################
# python3 $code_root/save_images.py
# python3 $code_root/identify_bulk.py
# ls -d post_process_bulk_gas/*/*_*/ > image_list
# python3 $code_root/identify_gas.py ${path_dat}

################################################################################
#                                   Part. 4                                    #
################################################################################
python3 $code_root/get_rxn_species.py
