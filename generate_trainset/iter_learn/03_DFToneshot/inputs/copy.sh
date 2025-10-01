cwd=`pwd`
for species in CF CF3 CH2F CHF2;do
    for ion_e in 10 30;do
        dst="${species}_${ion_e}"
        cp INCAR KPOINTS batch_nurion.j run.sh ${dst}
        cd ${dst}

        for i in {1..10};do
            dst="run_${i}"
            mkdir -p ${dst}/poscars
            cp INCAR KPOINTS batch_nurion.j ${dst}
        done

        count=0
        for j in `seq 1 1 50`;do
            src=${j}/poscars
            for poscar in `ls ${src}`;do
                count=$((count + 1))
                poscar_num=`echo ${poscar} | sed "s/POSCAR_//g"`
                cp ${src}/${poscar} run_${count}/poscars/POSCAR_${j}_${poscar_num}

                if (( ${count} == 10 ));then
                    count=0
                fi
            done
        done

        echo ${dst} Done
        cd ${cwd}
    done
done
