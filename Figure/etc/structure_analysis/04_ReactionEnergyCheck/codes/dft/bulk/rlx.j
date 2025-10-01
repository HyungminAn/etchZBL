#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="relax"       # Job name
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

function run_VASP() {
    local vasp_bin=$1
    mpiexec.hydra -np $SLURM_NTASKS ${vasp_bin}  >& stdout.x
}

function check_convergence() {
    local t=`grep 'stopping structural energy minimisation' OUTCAR -c`
    if (( `echo "$t == 1" | bc` )); then
        local t2=`grep 'free  ' OUTCAR -c`
        if ((`echo "$t2 == 1" | bc` )); then
            return 1
        fi
    fi
    return 0
}

function copy_previous_results() {
    local k=$1

    if (( `echo "$k / 10 " | bc` == 0 ));then
        local text="000"$k
    elif [ `echo "$k / 100 " | bc` == 0 ]; then
        local text="00"$k
    elif [ `echo "$k / 1000 " | bc` == 0 ]; then
        local text="0"$k
    else
        local text=$k
    fi

    mv POSCAR "POSCAR_"$text -f
    mv OUTCAR "OUTCAR_"$text -f
    mv XDATCAR "XDATCAR_"$text -f
    cp CONTCAR POSCAR -f
    mv stdout.x "std_"$text -f
}

function relax() {
    local k=0; local kmax=100;
    local vasp_bin=$1
    run_VASP ${vasp_bin}

    while [ $k -lt  $kmax ]
    do
        # run check_convergence and if success then break
        # if not then continue
        check_convergence
        if [ $? -eq 1 ]; then break
        fi
        copy_previous_results $k

        run_VASP ${vasp_bin}

        ((k=k+1))
    done
}

function make_POTCAR() {
    local path_POSCAR=$1
    local elements=$(sed -n 6p $path_POSCAR)
    for element in ${elements[@]}; do
        cat $path_potcar_dir/$element/POTCAR >> POTCAR
    done
}

function find_factors() {
    local num=$1
    local factors=()

    # Loop from 1 to the number itself
    for ((i = 1; i <= num; i++)); do
        # Check if 'i' divides 'num' evenly
        if [ $(($num % $i)) -eq 0 ]; then
            factors+=($i)
        fi
    done

    # Print the factors using a loop
    local result=""
    for factor in "${factors[@]}"; do
        result+=" $factor"
    done

    # Return the result
    echo "$result"
}

function npar_test() {
    local vasp_bin=$1

    local loop_time_min=10000
    for npar in $(find_factors $SLURM_NTASKS);do
        local dst="npar_test/${npar}"

        mkdir -p $dst
        cp POSCAR POTCAR KPOINTS ${dst}
        sed '/NPAR/s/2/'$npar'/g;
             /NSW/s/1000/0/g' INCAR > ${dst}/INCAR

        local cwd=$(pwd)
        cd ${dst}
        run_VASP ${vasp_bin}
        cd ${cwd}

        local loop_time=$(grep 'LOOP+' ${dst}/OUTCAR | awk '{print $7}')
        if [ $(echo "${loop_time} < ${loop_time_min}" | bc -l) -eq 1 ]; then
            local loop_time_min=${loop_time}
            local npar_min=${npar}
        fi
    done

    echo "${npar_min}"
}

function main() {
    ### Set VASP paths...
    local vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
    local vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'

    ### Run tight_relax
    relax ${vasp_std}
}

cwd=$(pwd)
for i in $(ls -d */);do
    cd ${i}
    main
    cd ${cwd}
done
