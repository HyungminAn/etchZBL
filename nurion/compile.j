cmd="cmake -C ../cmake/presets/nolib.cmake -C ../cmake/presets/oneapi_my_avx512_knl.cmake"
cmd="$cmd -D PKG_INTEL=yes"
cmd="$cmd -D BUILD_MPI=yes"
cmd="$cmd -D INTEL_ARCH=cpu"
cmd="$cmd -DCMAKE_BUILD_TYPE=Release"
cmd="$cmd -D PKG_MPI=on"
cmd="$cmd ../cmake/"
${cmd}

cmake --build . -j 68
