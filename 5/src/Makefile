include ../make.inc

SOURCE = hw5.c
OBJECTS = $(SOURCE:.c=.o)
EXECUTABLE=hw5

LIBS = -lpapi

all: $(EXECUTABLE)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS) -mkl

%.o : %.c
	$(CC) $(CFLAGS) $< -o $@ -mkl

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

purge: clean
	rm -f  *~

