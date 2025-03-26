gen_potcar="/home/andynn/codes/vasp/generate_potcar.sh"
cwd=$(pwd)
for i in $(ls -d */);do
    cd ${i}
    if [ -f "CONTCAR" ];then
        mv CONTCAR POSCAR
        echo ${i} "CONTCAR -> POSCAR"
    fi
    if [ ! -f "POTCAR" ];then
        sh ${gen_potcar} POSCAR > POTCAR
    fi
    for file in INCAR KPOINTS;do
        cp ${cwd}/C_g/${file} .
    done
    cd ${cwd}
done
