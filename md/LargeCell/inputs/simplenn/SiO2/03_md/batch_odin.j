#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="change_Temp_z_100eV"
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# Start measuring time
start_time=$(date +%s)

################## Main Code ##################
lmp_path="/home/andynn/lammps/build_etch_d2_loki34/lmp"
lammps_input="largecell_lmp.in"

path_code="/data2/andynn/Etch/05_md_nnp_ver_2/00_Codes"
path_rm_product="${path_code}/rm_product_graph.py"
path_add_slab="${path_code}/add_slab.py"
path_get_slab_z="${path_code}/find_slab_z.py"

path_base_structure="/data2/andynn/Etch/05_md_nnp_ver_2/01_AmorphousSlab/01_Bulk/FINAL.coo"

nions=800
run_cmd_base="mpirun -np $SLURM_NTASKS $lmp_path"
run_cmd_base="${run_cmd_base} -in ${lammps_input}"

for i in `seq 1 1 ${nions}`;do
    input_structure="CHF_shoot_"$((i - 1))".coo"

    # Check whether to add slab
    python3 ${path_add_slab} -i ${input_structure} -o copied_${input_structure} -b ${path_base_structure} -f 2 -t 10 -c 12

    if [ $? -ne 0 ]; then
        echo "Error occurred in slab addition"
        exit 1
    fi

    if [ -e "copied_${input_structure}" ];then
        mv copied_${input_structure} ${input_structure}
    fi
    crit_z=`python3 ${path_get_slab_z} ${input_structure}`
    if [ `echo "${crit_z} > 30.0" | bc` -eq "1" ];then
        temp_z=`echo "${crit_z}-20" | bc -l`
    else
        temp_z=10
    fi

    output_structure="CHF_shoot_"${i}".coo"
    path_log="log_"${i}".lammps"
    path_output="lammps_"${i}".out"

    run_cmd="${run_cmd_base} -var SEEDS $RANDOM"
    run_cmd="${run_cmd} -var input_structure ${input_structure}"
    run_cmd="${run_cmd} -var output_structure ${output_structure}"
    run_cmd="${run_cmd} -var i ${i} -l ${path_log}"
    run_cmd="${run_cmd} -var crit_z ${crit_z}"
    run_cmd="${run_cmd} -var temp_z ${temp_z}"

    echo "--- run ${i} start"
    ${run_cmd}
    echo "--- run ${i} end"
    # python3 ${path_rm_product} ${output_structure} ${i}
done
################## Main Code ##################

# End measuring time
end_time=$(date +%s)

# Calculate and display the elapsed time
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
