#!/bin/bash

NUMTHREADS=(2 4)
EXEC=bt.A.x
EXEC_DIR=/home/costrouc/class/cs594/hw/8/NPB3.3-OMP/bin
SAMPLEFREQ=100000

mkdir 3result

for NUMTHREAD in "${NUMTHREADS[@]}"
do
    export OMP_NUM_THREADS=${NUMTHREAD}
    
    mkdir 3result/${NUMTHREAD}/
    
    hpcrun -e PAPI_TOT_CYC@${SAMPLEFREQ} \
	-o 3result/${NUMTHREAD}/measurements \
	${EXEC_DIR}/${EXEC} >> 3result/${NUMTHREAD}/results3.txt

    hpcstruct ${EXEC_DIR}/${EXEC} \
	-o 3result/${NUMTHREAD}/${EXEC}.hpcstruct

    hpcprof -M thread \
	-S 3result/${NUMTHREAD}/${EXEC}.hpcstruct \
	3result/${NUMTHREAD}/measurements \
	-o 3result/${NUMTHREAD}/database 
done
