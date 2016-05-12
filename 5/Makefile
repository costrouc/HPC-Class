include make.inc

build: 
	$(MAKE) -C src/	

.PHONY: clean
clean:
	rm -f data/*
	$(MAKE) -C src/ clean

test: clean  build 
	./util/download-mat.sh
	./src/hw5 CRS 1 data/matrix.output.crs 
	./src/hw5 CCS 1 data/matrix.output.ccs
	./src/hw5 CRS 1 data/matrix.reorder.crs 
	./src/hw5 CCS 1 data/matrix.reorder.ccs
