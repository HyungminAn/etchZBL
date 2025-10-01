#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28         # Cores per node
#SBATCH --partition=loki4          # Partition name (skylake)
##
#SBATCH --job-name="gen_pt_zbl"
#SBATCH --time=02-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o STDOUT.%N.%j.out          # STDOUT, %N : nodename, %j : JobID
#SBATCH -e STDERR.%N.%j.err          # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### Set VASP paths...
vasp_std='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.std.x'
vasp_gam='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.gam.x'
vasp_ncl='/TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.vtst.ncl.x'

### To choose GPU nodes, turn on the option below...
# export CUDA_VISIBLE_DEVICES= 0

python get_residual_new_parallel.py
