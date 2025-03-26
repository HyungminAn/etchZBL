#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="genposcar"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

src="/data2/andynn/ZBL_modify/zbl_chgTot/02_NNP_RIE"
if [ -z ${src} ];then
    echo "Usage: sh copy.sh [src]"
    exit 1
fi

path_genPOSCAR="/data2/andynn/ZBL_modify/codes/SmallCell/03_DFToneshot/read_write_mod.py"
path_inputs="/data2/andynn/ZBL_modify/codes/SmallCell/03_DFToneshot/inputs"
cwd=$(pwd)
for species in $(ls -d ${src}/*/);do
    species=$(basename ${species})
    for ion_e in $(ls -d ${src}/${species}/*/);do
        ion_e=$(basename ${ion_e})

        dst="${species}_${ion_e}"
        python ${path_genPOSCAR} ${src}/${species}/${ion_e}
        mv structures ${dst}

        for file in INCAR KPOINTS batch_nurion.j run.sh;do
            cp ${path_inputs}/${file} ${dst}
        done

        for i in {1..10};do
            dst2="${dst}/run_${i}"
            mkdir -p ${dst2}/poscars
            for file in INCAR KPOINTS batch_nurion.j;do
                cp ${dst}/${file} ${dst2}
            done
        done

        count=0
        for j in $(seq 1 1 25);do
            for poscar in $(ls ${dst}/${j}/POSCAR_*);do
                count=$((count + 1))
                dst3="${dst}/run_${count}/poscars"

                poscar_num=$(basename ${poscar} | sed "s/POSCAR_//g")
                cp ${poscar} ${dst3}/POSCAR_${j}_${poscar_num}

                if (( ${count} == 10 ));then
                    count=0
                fi
            done
        done

        echo ${dst} Done
        cd ${cwd}
    done
done
