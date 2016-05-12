#include "hw2.h"

#include "omp.h"

#include <math.h>

/* result := result - matrixa x matrixb */
void dgemm(unsigned int m, unsigned int n, unsigned int l, double *matrixa, double *matrixb, double *result)
{
  unsigned int i, j, k;

#pragma omp for 
  for ( i=0; i<m; i++ )
    for( j=0; j<l; j++ )
      for( k=0; k<n; k++ )
	result[i * l + j] -= matrixa[i * n + k] * matrixb[k * l + j];

}

/* Solves trida * x = matrixa. Assumes that tridiagonal is lower triangular
 result is stored in marixa*/
void dtrsm(unsigned int m, unsigned int n, double *trida, double *matrixb)
{
  {
    unsigned int i, j, k;

#pragma omp for
    for( j=0; j<n; j++ )
      for ( i=0; i<m; i++ )
	{
	  for( k=0; k<i; k++ )
	    {
	      matrixb[i * n + j] -= trida[i * m + k ] * matrixb[k * n + j];
	    }
	  matrixb[i * n + j] /= trida[i * m + i];
	}
  }
}
