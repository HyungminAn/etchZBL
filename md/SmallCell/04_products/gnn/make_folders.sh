src="/data2/andynn/ZBL_modify/SmallCell/04_products/extract_poscars/poscars_classified/lammps"
for i in $(ls -d ${src}/*/);do
    first_file=$(ls -p ${i} | grep -v / | head -n 1)
    species=$(echo "$first_file" | awk -F_ '{print $NF}')
    mkdir -p ${species}
    cp ${i}/${first_file} ${species}/input.data
    echo "copy ${i}/${first_file} to ${species}/input.data" >> log

    for file in lammps.in;do
        cp ${file} ${species}
    done

    echo ${species} Done
done
