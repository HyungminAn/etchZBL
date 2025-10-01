export TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0;8.6;8.9;9.0"
DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
mkdir -p build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=${DCMAKE_PREFIX_PATH} -D PKG_COMPRESS=ON
make -j 4
