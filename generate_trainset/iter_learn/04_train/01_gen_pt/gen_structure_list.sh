src="/data2/andynn/Etch/05_iterative/Iter_1/03_dft_oneshot"
for i in CF CF3 CH2F CHF2;do
    for j in 10 30;do
        echo [ Iter_1_${i}_${j}eV ] >> structure_list

        for m in POSCAR_{1..50}_{0..4};do
            file_path="${src}/${i}_${j}/${m}/OUTCAR"
            echo "${file_path} :" >> structure_list

            if [ ! -f ${file_path} ];then
                echo Does not Exist: ${file_path}
            fi
        done
    done
done
