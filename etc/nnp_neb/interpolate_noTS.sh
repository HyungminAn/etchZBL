# Usage : sh -.sh *POSCAR_IS* *POSCAR_FS* *nimages*
path_code="/data2/andynn/SiN_etch/00_codes"
path_pos2lammps="${path_code}/my_pos2lammps.sh"
nimages=$3
n_totalimages=`echo "scale=0; ${nimages}+2" | bc`
output="${n_totalimages}images"

mkdir ${output}
cp $1 ${output}/POSCAR_IS
cp $2 ${output}/POSCAR_FS

cd ${output}
nebmake.pl POSCAR_IS POSCAR_FS ${nimages}
idx=0
for i in `find . -type d -name "[0-9][0-9]" | sort`;do
    ${path_pos2lammps} ${i}/POSCAR | awk '
        NR == 1 { print $0; }
        NR == 3 { print $1; }
        NR >= 14 { printf("%d %15f %15f %15f\n", $1, $3, $4, $5);}
    ' > coords.initial.${idx}

    idx=$((idx+1))
done

find . -depth -name "[0-9][0-9]" -type d -exec rm -r "{}" \;
${path_pos2lammps} POSCAR_IS > initial.dat
