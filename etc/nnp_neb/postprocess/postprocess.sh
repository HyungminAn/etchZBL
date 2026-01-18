path_code="/data2/andynn/SiN_etch/00_codes/neb/postprocess"
path_getbarrier=${path_code}/get_barrier_E.py
path_nebfinal=${path_code}/neb_final.py

cwd=`pwd`
for i in `ls -d */`;do
    cd ${i}/28images
    python ${path_getbarrier} log.lammps
    python ${path_nebfinal} initial.dat dump_{0..27}.lammps
    echo ${i} Done
    cd ${cwd}
done
