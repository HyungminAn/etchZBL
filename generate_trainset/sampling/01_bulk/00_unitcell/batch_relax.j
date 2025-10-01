#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4          # Partition name (skylake)
##
#SBATCH --job-name="relax"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'

### To choose GPU nodes, turn on the option below...
# export CUDA_VISIBLE_DEVICES= 0

src=`pwd`
for i in `ls -d */`;do
    cd ${i}

    mkdir entest/
    mv {300..700..50} entest/

    mkdir relax/
    cp POSCAR KPOINTS INCAR ${src}/POTCAR relax/

    cd relax/

################### rlx_batch_new.j ###################
    k=0
    kmax=100
    mpiexec.hydra -np $SLURM_NTASKS ${vasp_std}  >& stdout.x

    while [ $k -lt  $kmax ]
    do
        t=`grep 'stopping structural energy minimisation' OUTCAR -c`
        if (( `echo "$t <= 5" | bc` )); then
            t2=`grep 'free  ' OUTCAR -c`
            if ((`echo "$t2 <= 5" | bc` )); then
                break
            fi
        fi

        if (( `echo "$k / 10 " | bc` == 0 ));then
            text="000"$k
        elif [ `echo "$k / 100 " | bc` == 0 ]; then
            text="00"$k
        elif [ `echo "$k / 1000 " | bc` == 0 ]; then
            text="0"$k
        else
            text=$k
        fi

        mv POSCAR "POSCAR_"$text -f
        mv OUTCAR "OUTCAR_"$text -f
        mv XDATCAR "XDATCAR_"$text -f
        cp CONTCAR POSCAR -f
        mv stdout.x "std_"$text -f

        mpiexec.hydra -np $SLURM_NTASKS ${vasp_std}  >& stdout.x

        ((k=k+1))
    done
################### rlx_batch_new.j ###################

    cd ..

    cd ..
done
