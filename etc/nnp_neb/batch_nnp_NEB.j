#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=andynn        # Partition name (skylake)
##
#SBATCH --job-name="lammps_neb"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

path_lammps='/home/andynn/lammps/build_etch_d2_replica/lmp'
path_code="/data2/andynn/SiN_etch/00_codes/neb"
path_write_input="${path_code}/write_lammps_input.py"

########## run nnp_NEB ##########
path_interpolate="${path_code}/interpolate_noTS.sh"
sh ${path_interpolate} POSCAR_initial POSCAR_final 26

python ${path_write_input} POSCAR_initial
cp "neb.in" 28images/

cd 28images/
path_input='neb.in'

mpirun -np $SLURM_NTASKS ${path_lammps} -partition 28x1 -in ${path_input}
cd ..
