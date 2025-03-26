#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=ice          # Partition name (skylake)
##
#SBATCH --job-name="test"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'

### To choose GPU nodes, turn on the option below...
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1

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

python ./dump_cluster_analyis.py
