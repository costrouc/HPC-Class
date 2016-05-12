#include "hw9.h"
#include "gemm_optimize.h"

#include <papi.h>
#include <cblas.h>
#include <lapacke.h>

#include "xmmintrin.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_CACHE_SIZE 6144 * 1024 //Bytes

struct test_functions_t {
  void (*gemm) (int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
  char *name;
} test_functions [] = { {*ijk, "ijk"},
			{*ijk, "ikj"},
			{*ijk, "jik"},
			{*ijk, "jki"},
			{*ijk, "kij"},
			{*ijk, "kji"},
			{*dgemm_1, "dgemm_1"},
			{*dgemm_2, "dgemm_2"},
			{*dgemm_3, "dgemm_3"},
			{*dgemm_4, "dgemm_4"}
};

int num_test_functions = sizeof(test_functions) / sizeof(struct test_functions_t);


int matrix_sizes [] = { 40, 80, 120, 160, 200,
			240, 280, 320, 360, 400,
			440, 480, 520, 560, 600,
			640, 680, 720, 760, 800,
			840, 880, 920, 960, 1000};

int num_matrix_sizes = sizeof(matrix_sizes) / sizeof(int);

void flush_cache() {
  volatile char dummy[MAX_CACHE_SIZE];
  volatile int i;
  for ( i=0; i<MAX_CACHE_SIZE; i++ )
    dummy[i] = (char) rand();
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


int main(void)
{
  int i, j;
  
  /* Seed random number */
  srand(time(NULL));

  int event_set = papi_init();
  
  printf("Size\tMethod\tReal Time(s)\tFlops/Cycle\tMFLOPS\tRelative Error\n");
  
  for ( i=0; i<num_matrix_sizes; i++ )
    {

      int m, n, k;
      m = matrix_sizes[i];
      n = matrix_sizes[i];
      k = matrix_sizes[i];

      double *A, *B, *C, *Ctemp;
      
      A = (double *) _mm_malloc(sizeof(double) * m * n, 256);
      B = (double *) _mm_malloc(sizeof(double) * n * k, 256);
      C = (double *) _mm_malloc(sizeof(double) * m * k, 256);
      Ctemp = (double *) _mm_malloc(sizeof(double) * m * k, 256);

      long long start_values[2], end_values[2];
      long long start_usec, end_usec;
      long long total_flops, total_cycles;
      double total_usec;
      
      for ( j=0; j<num_test_functions; j++ )
	{
	  // Generate Random data (to nullify cache)
	  int l;
	  for ( l=0; l< m*n; l++ )  
	      A[l] = (double) rand() / RAND_MAX;
	  for ( l=0; l< n*k; l++ )
	      B[l] = (double) rand() / RAND_MAX;
	  for ( l=0; l< m*k; l++ )
	      Ctemp[l] = C[l] = (double) rand() / RAND_MAX;

	  flush_cache();
	  
	  // Test [ijk]{3} implemetations performance
	  start_usec = PAPI_get_real_usec();
	  PAPI_read(event_set, start_values);
	  test_functions[j].gemm(m, n, k, A, m, B, k, C, m);
	  PAPI_read(event_set, end_values);
	  end_usec = PAPI_get_real_usec();

	  total_usec = 1.0 * (end_usec - start_usec);
	  total_flops = (end_values[1] - start_values[1]);
	  total_cycles = (end_values[0] - start_values[0]);
	  
	  printf("%d\t%s\t", m, test_functions[j].name);
	  printf("%e\t%f\t%f\t", total_usec / 100000.0, total_flops / (1.0 * total_cycles), total_flops / total_usec);

	  //Verify the correctness of each [ijk]{3} implementation
	  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		      m, n, k, 1.0, A, m, B, n, 1.0, Ctemp, m);
	  for ( l=0; l< m*k; l++ )  
	    C[l] = C[l] - Ctemp[l]; //Cgemm - Catlas

	  double norm_Cdif = LAPACKE_dlange( LAPACK_COL_MAJOR, 'F', m, k, C, m);
	  double norm_Catlas = LAPACKE_dlange( LAPACK_COL_MAJOR, 'F', m, k, Ctemp, m);

	  printf("%f\n", norm_Cdif / norm_Catlas / 5E-16);
	}

      // Generate Random data (to nullify cache)
      int l;
      for ( l=0; l< m*n; l++ )  
	A[l] = (double) rand() / RAND_MAX;
      for ( l=0; l< n*k; l++ )
	B[l] = (double) rand() / RAND_MAX;
      for ( l=0; l< m*k; l++ )
	C[l] = (double) rand() / RAND_MAX;

      flush_cache();
      
      // Test [atlas] dgemm implementation performance
      start_usec = PAPI_get_real_usec();
      PAPI_read(event_set, start_values);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		  m, n, k, 1.0, A, m, B, n, 1.0, C, m);
      PAPI_read(event_set, end_values);
      end_usec = PAPI_get_real_usec();

      total_usec = 1.0 * (end_usec - start_usec);
      total_flops = (end_values[1] - start_values[1]);
      total_cycles = (end_values[0] - start_values[0]);
	  
      printf("%d\t%s\t", m, "atlas");
      printf("%e\t%f\t%f\t", total_usec / 100000.0, total_flops / (1.0 * total_cycles), total_flops / total_usec);
      printf("%f\n", 0.0);
      
      _mm_free(A); _mm_free(B); _mm_free(C); _mm_free(Ctemp);
    }

  PAPI_shutdown();
  return 0;
}
