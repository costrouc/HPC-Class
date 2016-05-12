CC=gcc
CFLAGS= -O3 -Wall
INCPATH=-I/usr/local/atlas/include
LIBPATH=-L/usr/local/atlas/lib
LIBS=-lpapi -lcblas -llapacke -latlas -msse3 -msse4.1 -mavx

build:
	$(CC) $(CFLAGS)  src/main.c src/hw9.c src/gemm_optimize.c $(INCPATH) $(LIBPATH) -o hw9 $(LIBS)
plot: test
	Rscript util/plot.r
test: build
	./hw9 > data/results.txt
