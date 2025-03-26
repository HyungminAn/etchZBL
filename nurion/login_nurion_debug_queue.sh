qsub -I -l select=1:ncpus=68:ompthreads=1 -l walltime=12:00:00 -A lammps -q debug
