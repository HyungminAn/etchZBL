idx_start=1
idx_end=250

for ion_type in CF CF3 CH2F CHF2;do
    for ion_E in 10 30;do

        count=0
        for i in `seq ${idx_start} 1 ${idx_end}`;do
            filename="structures/POSCAR_${i}"
            nions=`sed -n '7p' ${filename} | tr ' ' '\n' | awk '{sum += $1} END {print sum}'`
            count=$((count + nions))
        done
        echo ${ion_type} ${ion_E} ${count}

        idx_start=$((idx_start + 250))
        idx_end=$((idx_end + 250))
    done
done
