path_dft_in="/data2/andynn/ZBL_modify/codes/SmallCell/05_reactionE/codes/dft_in"
src="$1"
if [ -z $src ]; then
    echo "Usage: $0 <src>"
    exit 1
fi

for i in $(ls -d ${src}/[0-9]*/*/); do
    dst="run/$(sed "s#${src}##g" <<< $i)"
    mkdir -p ${dst}

    cp ${i}/POSCAR ${dst}
    for file in INCAR KPOINTS;do
        cp ${path_dft_in}/${file} ${dst}
    done

    echo ${dst}
done

path_lmp2pos="/home/andynn/codes/preprocess/lmpdat2vasp.py"
for i in $(ls -d ${src}/gas/*/*/); do
    dst="run/$(sed "s#${src}##g" <<< $i)"
    mkdir -p ${dst}

    py ${path_lmp2pos} ${i}/coo ${dst}/POSCAR
    for file in INCAR KPOINTS;do
        cp ${path_dft_in}/${file} ${dst}
    done

    echo ${dst}
done

cd run/

total_run=10
count=0
for i in $(ls -d */); do
    dst="run_${count}"
    mkdir -p $dst
    mv $i $dst
    count=$((count+1))
    if [ $count -eq $total_run ]; then
        count=0
    fi
done

for i in $(ls -d run_*/); do
    cp ${path_dft_in}/batch_nurion.j ${i}
done
