include make.inc

all:
	@$(MAKE) -C $(SOURCE_DIR)
	mv src/hw4 .

.PHONY: clean purge
clean:
	rm -f hw3 *~
	@$(MAKE) -C $(SOURCE_DIR) clean

purge: clean
	rm -f *~
	@$(MAKE) -C $(SOURCE_DIR) purge

test:
	./hw4 > data/chol_qr.txt

release: purge
	cd ..; tar -zcvf hw4.tar.gz 4/; mv hw4.tar.gz 4/.
