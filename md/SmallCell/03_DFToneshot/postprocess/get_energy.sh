for ion_type in CF CF3 CH2F CHF2;do
    for ion_e in 10 30;do
        for incidence in {1..50};do
            for sample_freq in {0..4};do
                src="${ion_type}_${ion_e}/POSCAR_${incidence}_${sample_freq}/OUTCAR"
                e_dft=`grep 'free  ' ${src} | awk '{print $5}'`
                nions=`grep 'NIONS' ${src} | awk '{print $12}'`
                echo ${ion_type} ${ion_e} ${incidence} ${sample_freq} ${e_dft} ${nions}
            done
        done
    done
done
