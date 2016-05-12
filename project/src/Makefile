CC=cc
CFLAGS=-c -O2 -Wall -Wextra
LDFLAGS=
LIBS=-lm -lpapi
DEFINE=-DNDEBUG
SOURCES=main.c utils.c dist_matrix.c caps.c test.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=caps

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(DEFINE) -o $@ $(LIBS) -mkl

debug: DEFINE = -DDEBUG_CAPS
debug: CFLAGS += -g
debug: $(EXECUTABLE)

.c.o:
	$(CC) $(CFLAGS) $(DEFINE) $< -o $@ -mkl

test: $(EXECUTABLE)
	aprun -n 7 ./caps test 7 2 1

test_full: $(EXECUTABLE)
	aprun -n 343 ./caps test_full

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

purge: clean
	rm -f *~ 
