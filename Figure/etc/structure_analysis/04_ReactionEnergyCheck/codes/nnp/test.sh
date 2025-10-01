cwd=$(pwd)

for i in $(ls -d */);do
    cd ${i}
    name=$(basename ${i})
    if [[ ${name} == *"_s" ]];then
        lmp_input="${cwd}/lammps_s.in"
    else
        lmp_input="${cwd}/lammps.in"
    fi
    echo $(basename ${i}) ${lmp_input}
    cd ${cwd}
done
