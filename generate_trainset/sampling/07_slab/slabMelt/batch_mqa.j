#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4          # Partition name (skylake)
##
#SBATCH --job-name="slab_melt"
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'

WD1="anneal"
mkdir -p $WD1
cp KPOINTS POTCAR ./$WD1/
cp POSCAR ./$WD1/POSCAR
cd ./$WD1
cp ../INCAR_${WD1} ./INCAR
mpiexec.hydra -np $SLURM_NTASKS ${vasp_gam}  >& stdout.x
cd ../

WD2="melt"
mkdir -p $WD2
cp KPOINTS POTCAR ./$WD2/
cp ./$WD1/CONTCAR ./$WD2/POSCAR
cd ./$WD2
cp ../INCAR_${WD2} ./INCAR
mpiexec.hydra -np $SLURM_NTASKS ${vasp_gam}  >& stdout.x
cd ../

WD3="liquid"
mkdir -p $WD3
cp KPOINTS POTCAR ./$WD3/
cp ./$WD2/CONTCAR ./$WD3/POSCAR
cd ./$WD3
cp ../INCAR_${WD3} ./INCAR
mpiexec.hydra -np $SLURM_NTASKS ${vasp_gam}  >& stdout.x
cd ../
