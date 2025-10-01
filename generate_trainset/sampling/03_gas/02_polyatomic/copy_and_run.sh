for i in `ls -d */`;do
    cp KPOINTS batch_nurion.j ${i}
    cp INCAR_md ${i}/INCAR
done
