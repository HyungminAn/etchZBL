#!/bin/bash
module purge

module load craype-network-opa craype-mic-knl  intel/oneapi_21.3 impi/oneapi_21.3 

make yes-compress
make yes-manybody
make yes-extra-compute
make yes-extra-dump
make yes-extra-fix
make yes-opt
make yes-replica
make yes-intel

make intel_cpu_intelmpi -j 68 | tee log
