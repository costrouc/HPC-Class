#include "dist_matrix.h"
#include "utils.h"
#include "caps.h"

#include <papi.h>
#include <mpi.h>
#include <mkl.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double f1(int i, int j) {
  return i + j * 7;
}
double f2(int i, int j) {
  return rand() / (1.0 * RAND_MAX);
}
double f3(int i, int j) {
  return 1.0 + 0.0 * (i*j);
}
double f4(int i, int j) {
  return 0.0 * (i*j);
}
double f5(int i, int j) {
  return (i==j) ? 1.0 : 0.0;
}
double f6(int i, int j) {
  return (i <= j) ? 2.0 : 1.0;
}

struct test_parameters_t {
  int numProc;
  int s;
  int scale;
} test_parameters[] = {
  //{7, 2, 1}, {7, 2, 10}, {7, 2, 100},
  //{49, 2, 10}, {49, 2, 100}, {49, 2, 200}, {49, 4, 100}, {49, 4, 200}, {49, 4, 250}, {49, 4, 50},
  //{49, 5, 100}, {49, 2, 300}, {49, 3, 300}, {49, 3, 500}, 
  {343, 4, 10}, {343, 5, 15}, {343, 5, 20}, {343, 4, 25}, {343, 5, 30}, {343, 5, 35},
  {343, 4, 40}, {343, 5, 45}, {343, 5, 27}
};

int num_tests = sizeof(test_parameters) / sizeof(struct test_parameters_t);

void test_caps_full() {

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(time(NULL) * rank);

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    exitWithError("Failed to initialize PAPI...\n");
  
  if (rank == ROOT)
    printf("Full Test [%d tests]\n", num_tests);

  char filename[] = "../data/results.txt";
  FILE *test_output;
  if (rank == ROOT) {
    test_output = fopen(filename, "w");
    
    if (test_output == NULL)
      exitWithError("Failed to open file '../data/results.tct' - rank 0\n");

    fprintf(test_output, "n\tP\tbm\ts\tk\tscale\tp\tq\ttime\tflipins\tGFLOPS\n");
  }
    
  int i;
  for (i=0; i<num_tests; i++) {

    int
      numProcs = test_parameters[i].numProc,
      s = test_parameters[i].s,
      scale = test_parameters[i].scale;

    int k;
    int n;
    int bm, bn;
    int p, q;

    k = ilog7(numProcs);
    
    if (k % 2 == 0) {
      n = scale * ipow(2, s) * ipow(7, (k/2));
      bm = bn = scale;
      p = q = ipow(7, (k/2));
    } else {
      n = scale * ipow(2, s) * ipow(7, (k+1)/2);
      bm = scale; bn = scale*7;
      p = ipow(7, (k+1)/2); q = ipow(7, (k-1)/2);
    }

    if (rank == ROOT)
      if (size >= numProcs) {
	printf("[%2d ] numProcs: %d s: %d scale: %d n: %d\n", i, numProcs, s, scale, n);
      } else {
	printf("[%2d ] numProcs %d requested is more than available\n", i, numProcs);
	continue;
      }
    
    double max_local_memory_mb = 2 * 1024;

    MPI_Comm job_comm;
    MPI_Comm_split(MPI_COMM_WORLD, (rank < numProcs) ? 1 : 0, rank, &job_comm);

    int job_comm_size;
    MPI_Comm_size(job_comm, &job_comm_size);

    if (job_comm_size == numProcs) {
    
      struct dbcl_t a, b, c;
      dist_matrix_init(job_comm, &a, p, q, bm, bn, n, n, f2);
      dist_matrix_init(job_comm, &b, p, q, bm, bn, n, n, f2);
      dist_matrix_init(job_comm, &c, p, q, bm, bn, n, n, f4);

      // Run test
      long long caps_start_time, caps_end_time;
      caps_start_time = PAPI_get_real_usec();
      caps(&a, &b, &c, max_local_memory_mb);
      caps_end_time = PAPI_get_real_usec();

      double caps_total_time_sec = (caps_end_time - caps_start_time) / 1000000.0;
      double caps_total_flop_ins = 2.0*(a.m)*(a.m)*(a.n); 

      if (rank == ROOT) {
	fprintf(test_output, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t", n, numProcs, bm, s, k, scale, p, q);
	fprintf(test_output, "%3.2f\t%3.2f\t%f\n", caps_total_time_sec, caps_total_flop_ins, caps_total_flop_ins / (caps_total_time_sec * 1.0E9));
      }

      dist_matrix_free(&a);
      dist_matrix_free(&b);
      dist_matrix_free(&c);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == ROOT)
    fclose(test_output);
}

void test_caps_light(int numProcessors, int s, int scale) {

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(time(NULL) * rank);
  
  // Check Parameters
  int k;
  if ( (k = ilog7(size)) == -1 )
    exitWithError("For this implementation of CAPS we require # Processors to be 7^k");

  if ( s < k )
    exitWithError("Due to the folding it is required s >= k + l");

  int n;
  int bm, bn;
  int p, q;

  if (k % 2 == 0) {
    n = scale * ipow(2, s) * ipow(7, (k/2));
    bm = bn = scale;
    p = q = ipow(7, (k/2));
  } else {
    n = scale * ipow(2, s) * ipow(7, (k+1)/2);
    bm = scale; bn = scale*7;
    p = ipow(7, (k+1)/2); q = ipow(7, (k-1)/2);
  }

  double max_local_memory_mb = 2 * 1024;
  
  // Print Matrix Test Parameters
  if (rank == ROOT) {
    printf("Parameters: {s: %d, k: %d, scale: %d}\n", s, k, scale);
    printf("Local Memory per Processor: %2.2f MBytes\n", max_local_memory_mb);
    printf("Global Dimension: %d x %d\n", n, n);
    printf("Block Dimension: %d x %d\n", bm, bn);
    printf("Processor Dimension: %d x %d\n", p, q);
    printf("Group (P * B) size: %d x %d [Notice always square]\n\n", bm*p, bn*q);
  }

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  struct dbcl_t a, b, c;
  dist_matrix_init(comm, &a, p, q, bm, bn, n, n, f2);
  dist_matrix_init(comm, &b, p, q, bm, bn, n, n, f2);
  dist_matrix_init(comm, &c, p, q, bm, bn, n, n, f4);

  if (rank == ROOT)
    printf("Matrix A:\n");
  dist_matrix_print(&a);

  if (rank == ROOT)
    printf("Matrix B:\n");
  dist_matrix_print(&b);
  

  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    exitWithError("Failed to initialize PAPI...\n");

  // Run test
  long long caps_start_time, caps_end_time;
  caps_start_time = PAPI_get_real_usec();
  caps(&a, &b, &c, max_local_memory_mb);
  caps_end_time = PAPI_get_real_usec();

  if (rank == ROOT)
    printf("Matrix C:\n");
  dist_matrix_print(&c);
  
  // Print Results
  double caps_total_time_sec = (caps_end_time - caps_start_time) / 100000.0;
  double caps_total_flop_ins = 2.0*(a.m)*(a.m)*(a.n); 
  if (rank == ROOT) {
    printf("Total time: %3.2f [sec]\n", caps_total_time_sec);
    printf("Total floating point ins: %3.2e\n", caps_total_flop_ins);
    printf("Effective flops: %3.2f GFLOPS\n\n", caps_total_flop_ins / (caps_total_time_sec * 1.0E9));
    printf("Beginning PBLAS test:\n");
  }

#if defined USEPBLAS  	/* Currently this will segfault.. */
  //Check for Correctness with mkl pblas (and measure relative error)
  struct dbcl_t a_pblas, b_pblas;
  dist_matrix_deepcopy(&a, &a_pblas);
  dist_matrix_deepcopy(&b, &b_pblas);
  dist_matrix_init(comm, &c, p, q, bm, bn, n, n, f4);

  int blacs_ictxt;
  Cblacs_pinfo(&rank, &size);
  Cblacs_get( -1, 0, &blacs_ictxt);
  Cblacs_gridinit(&blacs_ictxt, 'C', a.p, a.q);
    
  int desca[9] = {1, blacs_ictxt, a.lM, a.lN, a.bm, a.bn, 0, 0, a.lM * a.bm};
  int descb[9] = {1, blacs_ictxt, b.lM, b.lN, b.bm, b.bn, 0, 0, b.lM * b.bm};
  int descc[9] = {1, blacs_ictxt, c.lM, c.lN, c.bm, c.bn, 0, 0, c.lM * c.bm};

  long long pblas_start_time, pblas_end_time;
  printf("Made it here\n");
  pblas_start_time = PAPI_get_real_usec();
  pdgemm_('N', 'N', a_pblas.m, a_pblas.n, b_pblas.n, 1.0,
	  a.value, 0, 0, desca,
	  b.value, 0, 0, descb,
	  0.0, c.value, 0, 0, descc);
  pblas_end_time = PAPI_get_real_usec();
  
  double pblas_total_time_sec = (pblas_end_time - pblas_start_time) / 1000000.0;
  double pblas_total_flop_ins = 2.0*(a_pblas.m)*(a_pblas.m)*(a_pblas.n); 
  if (rank == ROOT) {
    printf("Total time: %3.2f [sec]\n", pblas_total_time_sec);
    printf("Total floating point ins: %3.2e\n", pblas_total_flop_ins);
    printf("Effective flops: %3.2f GFLOPS\n", pblas_total_flop_ins / (pblas_total_time_sec * 1.0E9)); 
  }
  
  dist_matrix_free(&a_pblas);
  dist_matrix_free(&b_pblas);

  //Cblacs_barrier(blacs_ictxt,"A");
  Cblacs_gridexit(0);
#endif

  dist_matrix_free(&a);
  dist_matrix_free(&b);
  dist_matrix_free(&c);
}
