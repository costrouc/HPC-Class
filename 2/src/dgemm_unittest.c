#include "test.h"
#include "hw2.h"

#include <papi.h>
#include <omp.h>
#include <cblas.h>
#include <lapacke.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void dgemm_initialize(double **data, double **result, double **result_blas, unsigned int max_size)
{
  int retval = 0;
  
  if ((*data = (double *) malloc(sizeof(double) * max_size * max_size * 2)) == NULL)
    test_fail(__FILE__, __LINE__, "malloc", retval);
  
  if ((*result = (double *) malloc(sizeof(double) * max_size * max_size)) == NULL)
    test_fail(__FILE__, __LINE__, "malloc", retval);

  if ((*result_blas = (double *) malloc(sizeof(double) * max_size * max_size)) == NULL)
    test_fail(__FILE__, __LINE__, "malloc", retval);
      
  /* Initialize data with random numbers Uniform [0, 1] */
  unsigned int i;
  for ( i=0; i< (max_size * max_size * 2); i++)
    (*data)[i] = (double) rand() / RAND_MAX;
}

int dgemm_unittest(unsigned int min_size, unsigned int max_size, FILE *output_file) {
    double *data, *result, *result_blas;
    dgemm_initialize(&data, &result, &result_blas, max_size);

    int test_status = TEST_SUCCESS;
    
    unsigned char offseta;
    unsigned char offsetb;

    test_print_header(output_file);
    
#pragma omp parallel default(shared)
    {
      struct test_statistics_t test_stat;
      test_papi_initialize(&test_stat);

      int tid = omp_get_thread_num();
      
      unsigned int i, j;
      for ( i=min_size; i< max_size; i++ )
	{
	  /* reset the results */
#pragma omp single
	  {
	    for ( j=0; j<i*i; j++ )
	      result[j] = result_blas[j] = 0.0;
      
	    /* Select random start location of Matrix a and Matrix b. 1 is
	       chosen to ensure we do not exceed the size of the array*/
	    offseta = rand() % (max_size * max_size);
	    offsetb = rand() % (max_size * max_size);
	  }
	  
	  int retval;
      
	  test_stat.elapsed_time = PAPI_get_real_usec();
	  if ((retval = PAPI_reset(test_stat.event_set)) != PAPI_OK)
	    test_fail(__FILE__, __LINE__, "PAPI_reset", retval);
      
	  dgemm(i, i, i, data + offseta, data + offsetb, result);

	  if ((retval = PAPI_read(test_stat.event_set, test_stat.counters)) != PAPI_OK)
	    test_fail(__FILE__, __LINE__, "PAPI_read", retval);
	  test_stat.elapsed_time = PAPI_get_real_usec() - test_stat.elapsed_time;

#pragma omp critical
	  test_print_statistic(tid, i, test_stat, output_file);

#pragma omp single
	  {
	    /* Preform Matrix Matrix Multiplication using cblas*/
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
			i, i, i, -1.0, data + offseta, i, data + offsetb, i, 1.0, result_blas, i);

	    /* Calculate the Frobenius Norm for both matricies*/
	    double result_norm, result_blas_norm;
      
	    result_norm = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', i, i, result, i);
	    result_blas_norm = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', i, i, result_blas, i);
      
	    /* Check relative error */
	    if (abs(result_norm - result_blas_norm) > MACH_ERROR)
	      test_status = TEST_FAIL;
	  }
	}

      test_papi_destroy(&test_stat);

    }
  free(data); free(result); free(result_blas);

  return test_status;
}
