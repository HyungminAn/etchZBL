path_lmp="/home01/x3045a07/lammps/build_avx512_knl_recompile/lmp"
path_lmp_in="lammps_oneshot.in"
strin="input.data" 

#cmd="gdb --args mpirun -np 64 ${path_lmp}"
cmd="${path_lmp}"
cmd="${cmd} -v strin ${strin}"
cmd="${cmd} -v i 1"
cmd="${cmd} -in ${path_lmp_in}"
echo ${cmd}
${cmd}
