include ../make.inc

SOURCE = main.c hw4.c
OBJECTS = $(SOURCE:.c=.o)
EXECUTABLE=hw4

LIBS = -lpapi -lm -lblas -llapacke -lcublas

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

%.o : %.c
	$(CC) -g $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

purge: clean
	rm -f  *~
