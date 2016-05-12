include make.inc

all:
	@$(MAKE) -C $(SOURCE_DIR)
	mv src/hw3 .

plot:
	python utils/plot.py

.PHONY: clean purge
clean:
	rm -f hw3 *~
	@$(MAKE) -C $(SOURCE_DIR) clean

purge: clean
	rm -f *~
	@$(MAKE) -C $(SOURCE_DIR) purge

test:
	./hw3 > data/chol_qr.txt

release: purge
	cd ..; tar -zcvf hw3.tar.gz 3/; mv hw3.tar.gz 3/.
