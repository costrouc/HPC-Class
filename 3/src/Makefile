include ../make.inc

SOURCE = main.c hw3.c
OBJECTS = $(SOURCE:.c=.o)
EXECUTABLE=hw3

LIBS = -lpapi -lm -lblas -llapacke

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

purge: clean
	rm -f  *~
