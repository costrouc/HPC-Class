include make.inc

all:
	@$(MAKE) -C $(SOURCE_DIR)
	mv src/hw2 .

plot:
	python utils/plot.py

.PHONY: clean
clean:
	rm hw2 *~
	@(MAKE) -C $(SOURCE_DIR) clean

test:
	./hw2 1 1000

release: 
	cd ..; tar -zcvf hw2.tar.gz 2/; mv hw2.tar.gz 2/.
