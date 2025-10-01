#!/bin/sh
#PBS -V
#PBS -N vts
#PBS -A vasp
#PBS -q normal
#PBS -l select=1:ncpus=64:mpiprocs=64:ompthreads=1
#PBS -l walltime=48:00:00

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > nodefile
NPROC=`wc -l < $PBS_NODEFILE`

vasp_std='/home01/x2702a08/ex/vasp_std_knl_'
vasp_gam='/home01/x2702a08/ex/vasp_gam_knl_'

WD1="premelt"
mkdir -p $WD1
cp KPOINTS POTCAR ./$WD1/
cp POSCAR ./$WD1/POSCAR
cd ./$WD1
cp ../INCAR_premelt ./INCAR
mpirun -np $NPROC $vasp_gam  >& stdout.x
cd ../

WD2="melt"
mkdir -p $WD2
cp KPOINTS POTCAR ./$WD2/
cp ./$WD1/CONTCAR ./$WD2/POSCAR
cd ./$WD2
cp ../INCAR_melt ./INCAR
mpirun -np $NPROC $vasp_gam  >& stdout.x
cd ../

WD3="quench"
mkdir -p $WD3
cp KPOINTS POTCAR ./$WD3/
cp ./$WD2/CONTCAR ./$WD3/POSCAR
cd ./$WD3
cp ../INCAR_quench ./INCAR
mpirun -np $NPROC $vasp_gam  >& stdout.x
cd ../


WD4="anneal"
mkdir -p $WD4
cp KPOINTS POTCAR ./$WD4/
cp ./$WD3/CONTCAR ./$WD4/POSCAR
cd ./$WD4
cp ../INCAR_anneal ./INCAR
mpirun -np $NPROC $vasp_gam  >& stdout.x
cd ../
