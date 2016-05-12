#include "utils.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

void exitWithError(char *m) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == ROOT)
    fprintf(stderr, "Error: %s\n", m);

  exit(1);
}

int ilog7(int value) {
  int k = 0;
  while (value != 1) {
    if (value % 7 != 0)
      return -1;
    value /= 7;
    k++;
  }
  return k;
}

int ipow(int base, int exp) {
  int result = 1;
  while (exp)
    {
      if (exp & 1)
	result *= base;
      exp >>= 1;
      base *= base;
    }

  return result;
}
