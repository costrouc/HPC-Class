CC=gcc
CFLAGS=-c -Wall -O3
LDFLAGS=

SOURCEDIR = src
SOURCE = $(wildcard $(SOURCEDIR)/*.c)
OBJECTS = $(SOURCE:.c=.o)
EXECUTABLE=hw1

all: $(SOURCES) $(EXECUTABLE)

hw1: $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ -lpapi -lm -lblas

%.o : %.c
	$(CC) $(CFLAGS) $< -o $@

test: $(EXECUTABLE)
	./hw1 1 1000

plot:
	python utils/plot.py

clean:
	rm -f hw1 $(SOURCEDIR)/*.o hw1.tar.gz

clean_data:
	data/*.txt img/*.png

purge: clean
	rm -f  *~ $(SOURCEDIR)/*~

release: purge
	cd ..; tar -zcvf hw1.tar.gz 1/; mv hw1.tar.gz 1/.
