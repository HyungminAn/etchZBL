for i in CF CF3 CH2F CHF2;do
    for j in 10 30;do
        cd ${i}_${j}
        mv run_*/Done/POSCAR_* .
        rmdir run_*/Done
        rmdir run_*
        cd ..
        echo ${i}_${j} Done
    done
done
