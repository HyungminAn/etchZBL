#!/bin/sh
#PBS -V
#PBS -N Oneshot
#PBS -A vasp
#PBS -q normal
#PBS -l select=1:ncpus=64:mpiprocs=64:ompthreads=1
#PBS -l walltime=12:00:00

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > nodefile
NPROC=`wc -l < $PBS_NODEFILE`

vasp_std="/home01/$USER/vasp/vasp_std"
vasp_gam="/home01/$USER/vasp/vasp_gam"
path_potcar_dir="/home01/$USER/pot/"

mkdir -p Done
for i in $(ls poscars/ | sort -t_ -nk 2);do
    mkdir -p ${i}
    cp INCAR KPOINTS ${i}
    cp poscars/${i} ${i}/POSCAR

    # Generate POTCAR
    string_elements=$(head -n 6 poscars/${i} | tail -1)
    IFS=" "
    read -ra elements <<<  ${string_elements}
    for element in ${elements[@]};do
        cat ${path_potcar_dir}/${element}/POTCAR >> POTCAR
    done
    mv POTCAR ${i}

    cd ${i}
    mpirun -np $NPROC $vasp_gam  >& stdout.x
    cd ..

    mv ${i} Done/
done
