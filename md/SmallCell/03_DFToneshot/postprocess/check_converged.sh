for i in CF CF3 CH2F CHF2;do
    for j in 10 30;do
        cd ${i}_${j}
        n_folders=`ls | wc -l`
        echo "-----${i}_${j} (${n_folders})-----"
        for k in POSCAR_*;do
            is_unconverged=`grep 'converged' ${k}/OUTCAR | wc -l`
            if [ ${is_unconverged} != 0 ];then
                echo ${k}
            fi
        done
        cd ..
    done
done
