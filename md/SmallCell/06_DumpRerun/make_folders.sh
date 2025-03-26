dst="run"
src="/data2/andynn/ZBL_modify/SmallCell/02_NNP_RIE/log_20"
path_code="/data2/andynn/ZBL_modify/codes/SmallCell/06_DumpRerun/inputs/"
cwd=$(pwd)
for ion_type in CF CF3 CH2F;do
    for ion_E in 20 50;do
        mkdir -p ${dst}/${ion_type}/${ion_E}
        sed "s|my_src|${src}/${ion_type}/${ion_E}|g" ${path_code}/batch.j > ${dst}/${ion_type}/${ion_E}/batch.j
        cp ${path_code}/lammps.in ${dst}/${ion_type}/${ion_E}/lammps.in

        cd ${dst}/${ion_type}/${ion_E}
        sbatch batch.j
        cd ${cwd}
    done
done
