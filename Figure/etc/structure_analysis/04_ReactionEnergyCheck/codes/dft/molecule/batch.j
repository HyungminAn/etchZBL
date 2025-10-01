#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # Cores per node
#SBATCH --partition=csc2          # Partition name (skylake)
##
#SBATCH --job-name="gases"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'

cwd=$(pwd)
for i in $(ls -d */);do
    cd ${i}
    mpiexec.hydra -np $SLURM_NTASKS ${vasp_gam}  >& stdout.x
    cd ${cwd}
done
