all: cg

clean:
	rm hw6 *~
cg:
	nvcc -c -O3 cg.cu -o cg.o 
	nvcc cg.o -o hw6 -lcublas -lcuda
