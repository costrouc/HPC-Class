RELEASE_NAME=chris_ostrouchov_finalproject

build:
	$(MAKE) -C src

clean:
	rm -f $(RELEASE_NAME).tar.gz
	$(MAKE) -C src clean

test: 
	$(MAKE) -C src test

test_full: 
	$(MAKE) -C src test_full

plot:
	Rscript tools/plot.r

purge: clean
	rm -f *~
	$(MAKE) -C src purge

release: purge
	cd ..; tar -czvf $(RELEASE_NAME).tar.gz project; mv $(RELEASE_NAME).tar.gz project



