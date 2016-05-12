#!/bin/bash

NUMTHREADS=(1 2 4 8)

mkdir 2result

for NUMTHREAD in "${NUMTHREADS[@]}"
do

    export OMP_NUM_THREADS=${NUMTHREAD}
    /global/homes/c/costrouc/class/cs594/hw/8/NPB3.3-OMP/bin/bt.W.x >> 2result/results2.txt
done
