path_src="/data2/andynn/Etch/01_trainingset/oneshot"
for i in `ls -d */`;do
    cd ${i}

    for j in `ls -d */`;do
        cd ${j}

        cd oneshot
# python select_poscar.py ../OUTCAR
        cp ${path_src}/batch.j ./batch.j
        sbatch batch.j
        pwd
        cd ../

        cd ..
    done

    cd ..
done
