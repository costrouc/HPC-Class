include ../make.inc

SOURCE = dgemm_unittest.c dtrsm_unittest.c test.c main.c hw2.c
OBJECTS = $(SOURCE:.c=.o)
EXECUTABLE=hw2

LIBS = -lpapi -lm -lblas -llapacke

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS) -fopenmp

%.o : %.c
	$(CC) $(CFLAGS) $< -o $@ -fopenmp

clean:
	rm -f *.o

purge: clean
	rm -f  *~
