#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="nnp_oneshot"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'


lmp_path="/home/andynn/lammps/build_etch_d2_loki34/lmp"
cwd=$(pwd)
lmp_input="/data2_1/andynn/Etch/07_CFx_dissociation/inputs/oneshot/lammps.in"
src_dft="/data2_1/andynn/Etch/07_CFx_dissociation/inputs/oneshot/dft"

for i in $(ls -d */);do
    cd ${i}
    mkdir -p DFT
    cp POSCAR DFT/POSCAR
    cp ${src_dft}/{INCAR,KPOINTS,POTCAR} DFT
    cd DFT
    mpiexec.hydra -np $SLURM_NTASKS ${vasp_gam}  >& stdout.x
    cd ${cwd}
done
