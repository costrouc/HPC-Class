CC=gcc
CFLAGS= -O3 -Wall
INCPATH=-I/usr/local/atlas/include
LIBPATH=-L/usr/local/atlas/lib
LIBS=-lpapi -lcblas -llapacke -latlas -msse3 -msse4.1 -mavx

build:
	$(MAKE) -C src
	cp src/hw11 .
plot: 
	Rscript util/plot.r
test: build
	mpirun -np 4 hw11 > data/results.txt
