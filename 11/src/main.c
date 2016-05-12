#include "test.h"

#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

void exit_cleanup() {
  MPI_Finalize();
}

int main(int argc, char *argv[]) {
  
  MPI_Init (&argc, &argv);

  test_1();
  test_2();
  test_3();
  test_4();
  test_5();
  
  atexit(exit_cleanup);
  
  return 0;
}

