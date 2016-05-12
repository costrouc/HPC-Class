#include "test.h"
#include "hw11.h"

#include <mpi.h>
#include <papi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdlib.h>

void test_1() {

  int recv_message;
  
  int rank, size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  
  PassTokenComm(MPI_COMM_WORLD, &rank, &recv_message);

  assert(recv_message == ((rank-1 < 0) ? size - 1 : rank - 1));

  if (rank == 0) {
    fprintf(stderr, ">>> Part 1 of asignment complete\n");
    fprintf(stderr, ">>> TEST 1: PASS\n");
  }
}

void test_2() {
  
  int rank, size;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  assert( ((int) sqrt(size)) * ((int) sqrt(size)) == size); //Assert square  
  
  int recv_message;

  int m, n;
  m = n = sqrt(size);
  MPI_Comm comm_row, comm_col;

  CreateGemmCommGroups(m, n, MPI_COMM_WORLD, &comm_row, &comm_col);

  // Exchange token through row
  MPI_Comm_rank (comm_row, &rank);
  MPI_Comm_size (comm_row, &size);
  
  PassTokenComm(comm_row, &rank, &recv_message);
  assert(recv_message == ((rank-1 < 0) ? size - 1 : rank - 1));

  // Exchange token through column
  MPI_Comm_rank (comm_col, &rank);
  MPI_Comm_size (comm_col, &size);

  PassTokenComm(comm_col, &rank, &recv_message);
  assert(recv_message == ((rank-1 < 0) ? size - 1 : rank - 1));

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    fprintf(stderr, ">>> Part 2 of asignment complete\n");
    fprintf(stderr, ">>> TEST 1: PASS\n");
  }
}

void test_3() {

  int rank, size;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  
  int send_row_message, send_col_message;
  int recv_row_message, recv_col_message;

  assert( ((int) sqrt(size)) * ((int) sqrt(size)) == size); //Assert square  

  int m, n;
  m = n = sqrt(size);
    
  MPI_Comm comm_row, comm_col;

  CreateGemmCommGroups(m, n, MPI_COMM_WORLD, &comm_row, &comm_col);

  send_row_message = send_col_message = rank;
  int token = rank;
  int row_comp = 0, col_comp = 0; 
  int i = 0;

  do {
    // Exchange token through row
    MPI_Comm_rank (comm_row, &rank);
    MPI_Comm_size (comm_row, &size);
    
    PassTokenComm(comm_row, &send_row_message, &recv_row_message);
    //assert(recv_message == ((rank-1 < 0) ? size - 1 : rank - 1));

    // Exchange token through column
    MPI_Comm_rank (comm_col, &rank);
    MPI_Comm_size (comm_col, &size);

    PassTokenComm(comm_col, &send_col_message, &recv_col_message);
    //assert(recv_message == ((rank-1 < 0) ? size - 1 : rank - 1));

    send_row_message = recv_row_message;
    send_col_message = recv_col_message;

    if (recv_row_message == token)
      row_comp = 1;
    if (recv_col_message == token)
      col_comp = 1;
    
    i++;
  } while (!row_comp || !col_comp);

  assert( i-1 == (m > n) ? m : n);

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    fprintf(stderr, ">>> Part 3 of asignment complete\n");
    fprintf(stderr, ">>> TEST 3: PASS\n");
  }
}

/* You can easily create matricies to test just change the fa, fb, and fc functions */
double fa(int i, int j) {
  return rand() / (double) RAND_MAX;
  //return (i <= j) ? 1.0 : 0.0;
  //return (i == j-1 || i == j+1) ? 1.0 : 0.0;
  //return 1.0;
}

double fb(int i, int j) {
  return rand() / (double) RAND_MAX;
  //return (i >= j) ? 1.0 : 0.0;
  //return (double) (100.0 * i + j);
  //return 1.0 * j;
}

double fc(int i, int j) {
  return 0.0;
}

void test_4() {
  int size, rank;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  assert( ((int) sqrt(size)) * ((int) sqrt(size)) == size); //Assert square
  
  int p, q;
  p = q = sqrt(size);
  
  int bm = 2*p, bn = 2*q, bk = 2*p;
  int gm = 4*p, gn = 4*q, gk = 4*p;

  double *a, *b, *c;
  a = malloc(sizeof(double) * (gm*gk) / (p*q));
  b = malloc(sizeof(double) * (gk*gn) / (p*q));
  c = malloc(sizeof(double) * (gm*gn) / (p*q));

  if (a == NULL || b == NULL || c == NULL) {
    fprintf(stderr, "Failed to malloc memory for arrays\n");
    exit(1);
  }
  
  CreateMatricies(MPI_COMM_WORLD, p,  q, bm, bn, bk, gm, gn, gk, a, b, c, fa, fb, fc);

  MPI_Comm comm_row, comm_col;
  CreateGemmCommGroups(p, q, MPI_COMM_WORLD, &comm_row, &comm_col);

  if (rank == 0) {
    printf("A Matrix:\n");
  }
  PrintMPIMatrix(MPI_COMM_WORLD, gm, gk, bm, bk, p, q, a);
  if (rank == 0) {
    printf("B Matrix:\n");
  }
  PrintMPIMatrix(MPI_COMM_WORLD, gk, gn, bk, bn, p, q, b);
  
  pdgemm(MPI_COMM_WORLD, p, q, bm, bn, bk, gm, gn, gk, a, b, c);

  if (rank == 0) {
    printf("C = A x B matrix:\n");
  }
  PrintMPIMatrix(MPI_COMM_WORLD, gm, gn, bm, bn, p, q, c);
  
  if (rank == 0) {
    fprintf(stderr, ">>> Part 4 of asignment complete\n");
    fprintf(stderr, ">>> TEST 4: PASS\n");
  }
}

int papi_init() {
  PAPI_library_init(PAPI_VER_CURRENT);
  
  int event_set = PAPI_NULL;
  PAPI_create_eventset(&event_set);

  int papi_events[] = {
    PAPI_TOT_CYC,
    PAPI_DP_OPS
  };
  int num_papi_events = sizeof(papi_events) / sizeof(char);
  
  PAPI_add_events(event_set, papi_events, num_papi_events);
  PAPI_start(event_set);

  return event_set;
}

/* Test the effect of blocking factor i will only be testing on 4 processors*/
void test_5() {
  int size, rank;
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  assert( ((int) sqrt(size)) * ((int) sqrt(size)) == size); //Assert square
  
  int p, q;
  p = q = sqrt(size);

  if (rank == 0)
    printf("Block Size\tReal Time(s)\tFlops/Cycle\tMFLOPS\n");

  int event_set = papi_init();
  
  long long start_values[2], end_values[2];
  long long start_usec, end_usec;
  long long total_flops, total_cycles;
  double total_usec;
  
  int i;
  int max_blocksize = 4096;
  for (i=256; i<=max_blocksize; i*=2) {
    int bm = i*p, bn = i*q, bk = i*p;
    int gm = max_blocksize*p, gn = max_blocksize*q, gk = max_blocksize*p;

    double *a, *b, *c;
    a = malloc(sizeof(double) * (gm*gk) / (p*q));
    b = malloc(sizeof(double) * (gk*gn) / (p*q));
    c = malloc(sizeof(double) * (gm*gn) / (p*q));

    if (a == NULL || b == NULL || c == NULL) {
      fprintf(stderr, "Failed to malloc memory for arrays\n");
      exit(1);
    }

    if (rank == 0)
      fprintf(stderr, "Block size %d\n", i);
    
    CreateMatricies(MPI_COMM_WORLD, p,  q, bm, bn, bk, gm, gn, gk, a, b, c, fa, fb, fc);

    MPI_Comm comm_row, comm_col;
    CreateGemmCommGroups(p, q, MPI_COMM_WORLD, &comm_row, &comm_col);

    start_usec = PAPI_get_real_usec();
    PAPI_read(event_set, start_values);
    pdgemm(MPI_COMM_WORLD, p, q, bm, bn, bk, gm, gn, gk, a, b, c);
    PAPI_read(event_set, end_values);
    end_usec = PAPI_get_real_usec();

    total_usec = 1.0 * (end_usec - start_usec);
    total_flops = (end_values[1] - start_values[1]);
    total_cycles = (end_values[0] - start_values[0]);

    if (rank == 0) {
      printf("%d\t", i);
      printf("%e\t%f\t%f\n", total_usec / 100000.0, total_flops / (1.0 * total_cycles) * size, total_flops / total_usec * size);
    }

    free(a); free(b); free(c);
  }
  
  if (rank == 0) {
    fprintf(stderr, ">>> Part 5 blocking speed on matrix complete\n");
    fprintf(stderr, ">>> TEST 5: PASS\n");
  }
}
