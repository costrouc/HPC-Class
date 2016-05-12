#!/bin/bash

NUMTHREADS=(1)
EXEC=bt.A.x
EXEC_DIR=/home/costrouc/class/cs594/hw/8/NPB3.3-OMP/bin
SAMPLEFREQ=100000

mkdir 1result

for NUMTHREAD in "${NUMTHREADS[@]}"
do
    export OMP_NUM_THREADS=${NUMTHREAD}
    
    mkdir 1result/${NUMTHREAD}/
    
    hpcrun -e PAPI_FP_INS@${SAMPLEFREQ} \
	-e PAPI_TOT_INS@${SAMPLEFREQ} \
	-e PAPI_TOT_CYC@${SAMPLEFREQ} \
	-o 1result/${NUMTHREAD}/measurements \
	${EXEC_DIR}/${EXEC} >> 1result/${NUMTHREAD}/results1.txt

    hpcstruct ${EXEC_DIR}/${EXEC} \
	-o 1result/${NUMTHREAD}/${EXEC}.hpcstruct

    hpcprof -S 1result/${NUMTHREAD}/${EXEC}.hpcstruct \
	1result/${NUMTHREAD}/measurements \
	-o 1result/${NUMTHREAD}/database 
done
