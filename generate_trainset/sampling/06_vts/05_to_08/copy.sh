cwd=`pwd`
for i in {01..02}*;do
    cd ${i}
    for j in {01..04}*;do
        cd ${j}
        # cp ${cwd}/INCAR* ${cwd}/{KPOINTS,POTCAR,batch_nurion.j} .
        cp ${cwd}/INCAR* .
        cd ..
    done
    cd ..
done
