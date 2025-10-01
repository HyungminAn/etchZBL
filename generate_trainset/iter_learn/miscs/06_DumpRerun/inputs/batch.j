#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Cores per node
#SBATCH --partition=gpu2          # Partition name (skylake)
##
#SBATCH --job-name="dump_rerun"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

# lmp_path="/TGM/Apps/ANACONDA/2022.05/envs/pub_sevenn/bin/lmp_sevenn"
lmp_path="/home/andynn/lammps_sevenn/build/lmp"
# path_potential="/data2/andynn/ZBL_modify/codes/pot/pot_7net_chg_tot_l3i3_zbl_a0.104.pt"
path_potential="/data2/andynn/ZBL_modify/codes/pot/old/old_pot.pt"
# path_potential="/data2/shared_data/pretrained/7net_chgTot/deployed_serial.pt"
lmp_input="lammps.in"

src="my_src"
for i in {1..50};do
    if [ ${i} == 1 ];then
        path_in_input="${src}/input.data"
    else
        path_in_input="${src}/ion_shoot_$((i-1)).coo"
    fi
    path_in_dump="${src}/dump_${i}.lammps"
    path_out_thermo="thermo_${i}.dat"
    path_out_dump="dump_${i}.lammps"

    cmd="${lmp_path} -in ${lmp_input}"
    # cmd="${cmd} -v SEEDS ${RANDOM}"
    cmd="${cmd} -v path_potential ${path_potential}"
    cmd="${cmd} -v path_in_input ${path_in_input}"
    cmd="${cmd} -v path_in_dump ${path_in_dump}"
    cmd="${cmd} -v path_out_thermo ${path_out_thermo}"
    cmd="${cmd} -v path_out_dump ${path_out_dump}"

    eval ${cmd}
done
