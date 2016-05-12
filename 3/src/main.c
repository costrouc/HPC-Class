#include "hw3.h"

#include <cblas.h>
#include <lapacke.h>
#include <papi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_REPEAT_TEST 10

struct test_matrix_config_t {
  long int m;
  long int n;
} test_matrix[] = {
  { 1000, 32 },
  { 2000, 32 },
  { 3000, 32 }
};


int num_tests = sizeof(test_matrix) / sizeof(struct test_matrix_config_t);

int main(int argc, char *argv[])
{
  
  /* Seed random number */
  srand(time(NULL));

  float real_start_time, proc_start_time, mflops_start;
  float real_end_time, proc_end_time, mflops_end;
  long long flpins_start, flpins_end;
  int retval;

  printf("Cholesky QR Factorization\n\n");
  
  int i, j;
  for ( i=0; i<num_tests; i++ )
    {
      double real_time_average = 0.0, proc_time_average = 0.0;
      double mflops_average = 0.0, error_QR_average = 0.0, error_QtQ_average = 0.0;
      long long flpins_average = 0;

      long int n, m;
      m = test_matrix[i].m;
      n = test_matrix[i].n;
        
      printf("MATRIX M=%ld\tN=%ld\n", m, n);
      printf("Iter\tReal_Time [s]\tProc_Time [s]\tFLOPS\tMFLOPS\t\tnorm(A - QR)\tnorm(I - Q'Q)\n");
      
      for ( j=0; j<NUM_REPEAT_TEST; j++ )
	{
	  
	  double *A = (double *) malloc(sizeof(double) * m * n);
	  double *Q = (double *) malloc(sizeof(double) * m * n);
	  double *R = (double *) malloc(sizeof(double) * m * n);

	  double *temp = (double *) malloc(sizeof(double) * m * n );
  
	  unsigned int i;
	  for ( i=0; i<m*n; i++ )
	    temp[i] = A[i] = (double) rand() / RAND_MAX;

	  
	  /* Setup PAPI library and begin collecting data from the counters */
	  if((retval=PAPI_flops( &real_start_time, &proc_start_time, &flpins_start, &mflops_start))<PAPI_OK)
	    return 1;
	  
	  dge_cholesky_qr(m, n, A, Q, R);

	  /* Collect the data into the variables passed in */
	  if((retval=PAPI_flops( &real_end_time, &proc_end_time, &flpins_end, &mflops_end))<PAPI_OK)
	    return 1;
	  

	  /* temp(A) = -Q*R + temp(A)*/
	  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		      m, n, n, -1.0, Q, m, R, n, 1.0, temp, m);
	  
	  /* Norm temp */
	  double norm_A_QR = LAPACKE_dlange( LAPACK_COL_MAJOR, 'F', m, n, temp, m);
	  
	  /* temp[n, n] = Q[m,n]'*Q[m,n] */
	  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		      n, n, m, 1.0, Q, m, Q, m, 0.0, temp, n);

	  long int k;
	  /* Subtract the Identity */
	  for ( k=0; k<n; k++ )
	    temp[k * n + k] = 1.0 - temp[k * n + k];

	  /* Norm temp */
	  double norm_I_QtQ = LAPACKE_dlange( LAPACK_COL_MAJOR, 'F', n, n, temp, n);
	  
	  /* print statistics to stdout */
	  printf("%d\t", j);
	  printf("%e\t%e\t", real_end_time - real_start_time, proc_end_time - proc_start_time);
	  printf("%lld\t%f\t%e\t%e\n", flpins_end - flpins_start, mflops_end, norm_A_QR, norm_I_QtQ);

	  /* Add statistics to tabulate statistics */
	  real_time_average += real_end_time - real_start_time;
	  proc_time_average += proc_end_time - proc_start_time;
	  flpins_average += flpins_end - flpins_start;
	  mflops_average += mflops_end;
	  error_QR_average += norm_A_QR;
	  error_QtQ_average += norm_I_QtQ;
	  
	  free(A); free(Q); free(R);
	}
      real_time_average /= NUM_REPEAT_TEST;
      proc_time_average /= NUM_REPEAT_TEST;
      flpins_average /= NUM_REPEAT_TEST;
      mflops_average /= NUM_REPEAT_TEST;
      error_QR_average /= NUM_REPEAT_TEST;
      error_QtQ_average /= NUM_REPEAT_TEST;
      
      printf("============================================================================================\n");
      printf("AVG:\t%e\t%e\t%lld\t%f\t%e\t%e\n\n", real_time_average, proc_time_average, flpins_average, mflops_average, error_QR_average, error_QtQ_average);
    }
  
  return 0;
}
