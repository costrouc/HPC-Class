#include "utils.h"

#include <stdio.h>
#include <string.h>

#include <mpi.h>

void usage(char *exec_name) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == ROOT) {
    printf("Usage: %s <option>\n", exec_name);
    printf("options - [test, test_full]\n");
    printf("test:\n");
    printf("       numProcs - number of processors to use (7^k)\n");
    printf("       s - number of times matrix can be folded\n         s >= log_7(# Processors)\n\n");
    printf("       scale - scaling of blocks in m, n\n");
  }

  exit(1);
}

void finalize(){
  MPI_Finalize();
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  atexit(finalize);

  if (argc > 1) {
    if (strcmp("test", argv[1]) == 0) {
        if ( argc != 5 )
	  usage(argv[0]);
	
	int numProcs, s, scale;
	numProcs = atoi(argv[2]);
	s = atoi(argv[3]);
	scale = atoi(argv[4]);
	  
	test_caps_light(numProcs, s, scale);
    } else if (strcmp("test_full", argv[1]) == 0) {
      test_caps_full();
    } else {
      usage(argv[0]);
    }
  } else
    usage(argv[0]);
  
  return 0;
}
