src="/data2/andynn/ZBL_modify/SmallCell/04_products/extract_poscars/poscars_classified/vasp"
for i in $(ls -d ${src}/*/);do
    first_file=$(ls -p ${i} | grep -v / | head -n 1)
    species=$(echo "$first_file" | awk -F_ '{print $NF}')
    mkdir -p ${species}
    cp ${i}/${first_file} ${species}/POSCAR
    echo "copy ${i}/${first_file} to ${species}/POSCAR" >> log

    for file in INCAR KPOINTS;do
        cp ${file} ${species}
    done

    sh ~/codes/vasp/generate_potcar.sh ${species}/POSCAR > ${species}/POTCAR
    echo ${species} Done
done
