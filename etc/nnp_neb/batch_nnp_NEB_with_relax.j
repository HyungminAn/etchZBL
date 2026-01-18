#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4        # Partition name (skylake)
##
#SBATCH --job-name="lammps_neb"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

path_lammps='/home/andynn/lammps/build_etch_d2_replica/lmp'
path_code="/data2/andynn/SiN_etch/00_codes"
path_write_input="${path_code}/neb/write_lammps_input.py"

src_1="POSCAR_initial"
src_2="POSCAR_final"

########## Relax       ##########
# result would be 'path_input'
python ${path_write_input} ${src_1}

# Convert to lammps data file and relax each image
path_input='relax.in'
path_pos2lammps="${path_code}/my_pos2lammps.sh"

${path_pos2lammps} ${src_1} > input.data
cp ${src_1} ./POSCAR_initial_before_rlx
mpirun -np $SLURM_NTASKS ${path_lammps} -in ${path_input}
mv log.lammps log_relax_initial.lammps
mv dump.lammps dump_relax_initial.lammps
mv FINAL.coo relaxed_initial.coo

${path_pos2lammps} ${src_2} > input.data
cp ${src_2} ./POSCAR_final_before_rlx
mpirun -np $SLURM_NTASKS ${path_lammps} -in ${path_input}
mv log.lammps log_relax_final.lammps
mv dump.lammps dump_relax_final.lammps
mv FINAL.coo relaxed_final.coo

# Convert each relaxed image into VASP POSCAR
path_lmpdat2vasp="${path_code}/lmpdat2vasp.py"
python ${path_lmpdat2vasp} relaxed_initial.coo POSCAR_initial
python ${path_lmpdat2vasp} relaxed_final.coo POSCAR_final
########## Relax       ##########


########## run nnp_NEB ##########
path_interpolate="${path_code}/neb/interpolate_noTS.sh"
sh ${path_interpolate} POSCAR_initial POSCAR_final 26

python ${path_write_input} POSCAR_initial
cp "neb.in" 28images/

cd 28images/
path_input='neb.in'

mpirun -np $SLURM_NTASKS ${path_lammps} -partition 28x1 -in ${path_input}
cd ..
########## run nnp_NEB ##########
