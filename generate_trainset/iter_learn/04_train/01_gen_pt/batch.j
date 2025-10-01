#!/bin/bash
#SBATCH --nodelist=n014
#SBATCH --ntasks-per-node=32       # Cores per node
#SBATCH --partition=gpu2               # Partition name (skylake)
##
#SBATCH --job-name="pt_gen"
#SBATCH --time=04-00:00              # Runtime limit: Day-HH:MM
#SBATCH -o %N.%j.out                 # STDOUT, %N : nodename, %j : JobID
#SBATCH -e %N.%j.err                 # STDERR, %N : nodename, %j : JobID
#SBATCH --mail-type=FAIL,TIME_LIMIT  # When mail is sent (BEGIN,END,FAIL,ALL,TIME_LIMIT,TIME_LIMIT_90,...)

### To choose GPU nodes, turn on the option below...
export CUDA_VISIBLE_DEVICES=0
python run.py
