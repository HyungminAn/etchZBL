#!/bin/bash
#SBATCH --nodelist=n008
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu               # Partition name (skylake)
#SBATCH --job-name="etchMDtest"
#SBATCH --time=02-00:00               # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out           # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err           # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT    # When mail is sent (BEGIN, END, FAIL, ALL, TIME_LIMIT, TIME_LIMIT_90,...)

# Function to set up variables
setup_variables() {
    # lmp_path="/home/andynn/lammps/build_etch_d2_loki34/lmp"
    # lmp_path="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn"
    lmp_path="/home/andynn/lammps_sevenn/build/lmp"

    lammps_input="./largecell_lmp.in"
    path_potential="/data2_1/team_etch/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"

    path_code="/data2/andynn/ZBL_modify/codes/LargeCell/scripts"
    path_rm_product="${path_code}/rm_product_graph.py"
    path_add_slab="${path_code}/add_slab.py"
    path_get_slab_z="${path_code}/find_slab_z.py"

    path_base_structure="./base_structure.coo"

    nions=1
    run_cmd_base="${lmp_path} -in ${lammps_input}"
}

# Function to handle the slab addition and check its results
process_slab() {
    local i="$1"
    local input_structure="CHF_shoot_$((i - 1))_after_removing.coo"

    # Check whether to add slab
    python3 "${path_add_slab}" -i "${input_structure}" -o "copied_${input_structure}" -b "${path_base_structure}" -f 2 -t 10 -c 12

    if [ $? -ne 0 ]; then
        echo "Error occurred in slab addition"
        exit 1
    fi

    if [ -e "copied_${input_structure}" ]; then
        mv "copied_${input_structure}" "${input_structure}"
    fi

    local crit_z=$(python3 "${path_get_slab_z}" "${input_structure}")

    if [ "$(echo "${crit_z} > 30.0" | bc)" -eq "1" ]; then
        local temp_z=$(echo "${crit_z} - 20" | bc -l)
    else
        local temp_z=10
    fi

    echo "${input_structure}" "${crit_z}" "${temp_z}"
}

# Function to run LAMMPS simulation
run_simulation() {
    local i="$1"
    local input_structure="$2"
    local crit_z="$3"
    local temp_z="$4"

    local output_structure="CHF_shoot_${i}.coo"
    local path_log="log_${i}.lammps"
    local path_output="lammps_${i}.out"

    local run_cmd="${run_cmd_base}"
    run_cmd="${run_cmd}"
    run_cmd="${run_cmd} -var SEEDS $RANDOM"
    run_cmd="${run_cmd} -var input_structure ${input_structure}"
    run_cmd="${run_cmd} -var output_structure ${output_structure}"
    run_cmd="${run_cmd} -var i ${i}"
    run_cmd="${run_cmd} -l ${path_log}"
    run_cmd="${run_cmd} -var crit_z ${crit_z}"
    run_cmd="${run_cmd} -var temp_z ${temp_z}"
    run_cmd="${run_cmd} -var path_potential ${path_potential}"

    echo "--- run ${i} start"
    eval "${run_cmd}"
    echo "--- run ${i} end"
    python3 ${path_rm_product} ${output_structure} ${i}
}

check_conda_env() {
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

main() {
    check_conda_env "graph-tool"

    # Start measuring time
    start_time=$(date +%s)

    ################## Main Code ##################
    setup_variables

    for i in $(seq 1 1 ${nions}); do
        IFS=" " read -r input_structure crit_z temp_z <<< "$(process_slab ${i} | tail -n 1)"
        echo "input_structure: ${input_structure}"
        echo "crit_z: ${crit_z}"
        echo "temp_z: ${temp_z}"

        run_simulation "${i}" "${input_structure}" "${crit_z}" "${temp_z}"
    done

    ################## Main Code ##################

    # End measuring time
    end_time=$(date +%s)

    # Calculate and display the elapsed time
    elapsed_time=$((end_time - start_time))
    echo "Elapsed time: $elapsed_time seconds"
}

main
