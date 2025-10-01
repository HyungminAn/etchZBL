grep 'free  ' lammps_{1..2000}.out | awk '{print $3}' > ../energy_nnp.dat
