#include "unittest_hw1.h"

#include "hw1.h"

#include <papi.h>
#include <cblas.h>

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#define TEST_FAIL 1
#define TEST_SUCCESS 0

#define MACH_ERROR 1E-12

void test_fail(char *file, int line, char *call, int retval){
  printf("%s\tFAILED\nLine # %d\n", file, line);
  if ( retval == PAPI_ESYS ) {
    char buf[128];
    memset( buf, '\0', sizeof(buf) );
    sprintf(buf, "System error in %s:", call );
    perror(buf);
  }
  else if ( retval > 0 ) {
    printf("Error calculating: %s\n", call );
  }
  else {
    printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
  }
  printf("\n");
  exit(1);
}

int test_norm(unsigned int min_size, unsigned int max_size, FILE *output_file) {

  double *data = (double *) malloc(sizeof(double) * max_size * max_size * 4);
  double result = 0.0;
  double result_blas = 0.0;

  if (data == NULL)
    return TEST_FAIL;
  
  unsigned int i;
  for ( i=0; i< (max_size * max_size * 4); i++)
    data[i] = (double) rand() / RAND_MAX;
  
  float real_start_time, proc_start_time, mflops_start;
  float real_end_time, proc_end_time, mflops_end;
  long long flpins_start, flpins_end;
  int retval;
  
  for ( i=min_size; i< max_size; i++ )
    {
      /* reset the results */
      result = 0.0;
      result_blas = 0.0;
      
      /* Select random start location of Matrix a and Matrix b. 3 is
	 chosen to ensure we do not exceed the size of the array*/
      unsigned char offseta = rand() % (max_size * max_size * 3);
      
      /* Setup PAPI library and begin collecting data from the counters */
      if((retval=PAPI_flops( &real_start_time, &proc_start_time, &flpins_start, &mflops_start))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
      
      norm2(data + offseta, &result, i);
      
      /* Collect the data into the variables passed in */
      if((retval=PAPI_flops( &real_end_time, &proc_end_time, &flpins_end, &mflops_end))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

      print_statistics(output_file, i, real_end_time - real_start_time, proc_end_time - proc_start_time, flpins_end - flpins_start, mflops_end);

      /* Test implemented norm with cblas implementation */
      result_blas = cblas_dnrm2 (i, data + offseta, 1);

      /* Check relative error */
      if ((result - result_blas) / result > MACH_ERROR || (result_blas - result) / result > MACH_ERROR)
	return TEST_FAIL;
      
    }
  
  free(data);

  PAPI_shutdown();

  return TEST_SUCCESS;
}

int test_matvecmult(unsigned int min_size, unsigned int max_size, FILE *output_file) {
  /* Initialize data 4 times larger than largest matrix
     This ensures cache misses on each iteration when a
     random start location is used*/
  
  double *data = (double *) malloc(sizeof(double) * max_size * max_size * 2);
  if (data == NULL)
    return TEST_FAIL;
  
  double *result = (double *) malloc(sizeof(double) * max_size);
  if (result == NULL)
    return TEST_FAIL;

  double *result_blas = (double *) malloc(sizeof(double) * max_size);
  if (result_blas == NULL)
    return TEST_FAIL;
  
  unsigned int i,j;
  for ( i=0; i< (max_size * max_size * 2); i++)
    data[i] = (double) rand() / RAND_MAX;

  float real_start_time, proc_start_time, mflops_start;
  float real_end_time, proc_end_time, mflops_end;
  long long flpins_start, flpins_end;
  int retval;
  
  for ( i = min_size; i< max_size; i++ )
    {
      /* reset result values each iteration */
      for ( j=0; j<i; j++ )
	result[j] = result_blas[j] = 0.0;
      
      /* Select random start location of Matrix a and Matrix b. 3 is
	 chosen to ensure we do not exceed the size of the array*/
      unsigned char offseta = rand() % (max_size * max_size * 3);
      unsigned char offsetb = rand() % (max_size * max_size * 3);
      
      /* Setup PAPI library and begin collecting data from the counters */
      if((retval=PAPI_flops( &real_start_time, &proc_start_time, &flpins_start, &mflops_start))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
      
      matvecmult(data + offseta, data + offsetb, result, i);
      
      /* Collect the data into the variables passed in */
      if((retval=PAPI_flops( &real_end_time, &proc_end_time, &flpins_end, &mflops_end))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

      print_statistics(output_file, i, real_end_time - real_start_time, proc_end_time - proc_start_time, flpins_end - flpins_start, mflops_end);

      /* Test implemented matrix vector product implementation with cblas */
      cblas_dgemv (CblasRowMajor, CblasNoTrans, i, i, 1.0, data + offseta, i, data + offsetb, 1, 0.0, result_blas, 1);

      /* Calculate error as the sum of the differences between the matricies */
      double result_relative_diff = 0.0;
      for ( j=0; j<i; j++ )
	result_relative_diff += (result[j] - result_blas[j]) / result[j];

      /* Check relative error */
      if (result_relative_diff > MACH_ERROR || result_relative_diff < -MACH_ERROR)
	return TEST_FAIL;
      
    }
  
  free(data); free(result); free(result_blas);

  PAPI_shutdown();

  return 0;
}

int test_matmatmult(unsigned int min_size, unsigned int max_size, FILE *output_file) {
  /* Initialize data 4 times larger than largest matrix
     This ensures cache misses on each iteration when a
     random start location is used*/
  
  double *data = (double *) malloc(sizeof(double) * max_size * max_size * 2);
  if (data == NULL)
    return TEST_FAIL;
  
  double *result = (double *) calloc(sizeof(double), max_size * max_size);
  if (result == NULL)
    return TEST_FAIL;

  double *result_blas = (double *) calloc(sizeof(double), max_size * max_size);
  if (result_blas == NULL)
    return TEST_FAIL;
  
  unsigned int i, j;
  for ( i=0; i< (max_size * max_size * 2); i++)
    data[i] = (double) rand() / RAND_MAX;

  float real_start_time, proc_start_time, mflops_start;
  float real_end_time, proc_end_time, mflops_end;
  long long flpins_start, flpins_end;
  int retval;
  
  for ( i = min_size; i< max_size; i++ )
    {
      /* reset result values each iteration */
      for ( j=0; j<i * i; j++ )
	result[j] = result_blas[j] = 0.0;
      
      /* Select random start location of Matrix a and Matrix b. 3 is
	 chosen to ensure we do not exceed the size of the array*/
      unsigned char offseta = rand() % (max_size * max_size);
      unsigned char offsetb = rand() % (max_size * max_size);
      
      /* Setup PAPI library and begin collecting data from the counters */
      if((retval=PAPI_flops( &real_start_time, &proc_start_time, &flpins_start, &mflops_start))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
      
      matmatmult(data + offseta, data + offsetb, result, i);
      
      /* Collect the data into the variables passed in */
      if((retval=PAPI_flops( &real_end_time, &proc_end_time, &flpins_end, &mflops_end))<PAPI_OK)
	test_fail(__FILE__, __LINE__, "PAPI_flops", retval);

      print_statistics(output_file, i, real_end_time - real_start_time, proc_end_time - proc_start_time, flpins_end - flpins_start, mflops_end);

      /* Preform Matrix Matrix Multiplication using cblas*/
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
		  i, i, i, 1.0, data + offseta, i, data + offsetb, i, 0.0, result_blas, i);

      /* Calculate error as the sum of the differences between the matricies */
      double result_relative_diff = 0.0;
      for ( j=0; j<i*i; j++ )
	result_relative_diff += (result[j] - result_blas[j]) / result[j];

      /* Check relative error */
      if (result_relative_diff > MACH_ERROR || result_relative_diff < -MACH_ERROR)
	return TEST_FAIL;
      
    }
  
  free(data); free(result);

  PAPI_shutdown();

  return 0;
}

/* Prints running statistics [5]
   Size of Problem | Real Run Time [us] | Processor Run Time [us]
   Floating Point Instructions | Floating Point Opperatfions / second [mflops]*/
void print_statistics
(FILE *output_file,
 unsigned int size,
 float real_time,
 float proc_time,
 long long flpins,
 float mflops)
{
  fprintf(output_file, "%d\t%f\t%f\t%lld\t%f\n",
	  size,
	  real_time,
	  proc_time,
	  flpins,
	  mflops);
}
