for i in `ls -d */`;do
    cd ${i}
    awk 'NF{print} !NF{exit}' POSCAR > POSCAR_new
    mv POSCAR_new POSCAR
    cd ..

    echo ${i} Done
done
