CC=mpicc
CFLAGS=-c -Wall
LDFLAGS=-I/usr/local/atlas/include -L/usr/local/atlas/lib
SOURCES=hw11.c main.c test.c
OBJECTS=$(SOURCES:.c=.o)
LIBS=-lm -lcblas -lpapi -latlas
EXECUTABLE=hw11

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
purge:
	rm -f *~

test: $(EXECUTABLE)
	mpirun -np 4 $(EXECUTABLE)