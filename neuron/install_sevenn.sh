#0. set the required modules
echo module load gcc/10.2.0 >> ~/.bashrc
echo module load mpi/openmpi-4.1.1 >> ~/.bashrc
echo module load cmake/3.26.2 >> ~/.bashrc
echo module load cuda/12.3 >> ~/.bashrc
echo module load git/2.35.1 >> ~/.bashrc
export MKLROOT="/apps/compiler/intel/19.1.2/mkl" >> ~/.bashrc
source ~/.bashrc

# also, you need to set conda environment
conda create -n "zbl"
export CONDA_ENVS_PATH=/scratch/$USER/.conda/envs
export CONDA_PKGS_DIRS=/scratch/$USER/.conda/pkgs

#1. install pytorch
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

#2. install sevenn
pip install sevenn --user
echo export PATH="/home01/e1448a08/.local/bin/:$PATH" >> ~/.bashrc

#3. install lammps
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_2Aug2023_update3 --depth=1
sevenn_patch_lammps ./lammps_sevenn {--d3}
