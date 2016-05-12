#include "hw1.h"

#include <math.h>
#include <stdlib.h>

void norm2(double * vectora, double * result, unsigned int n)
{
  size_t i;
  
  for ( i=0; i<n; i++ )
    *result += vectora[i] * vectora[i];

  *result =  sqrt( *result );
}

/* Matricies are assumed to be square */
void matvecmult(double *matrixa, double *vectora, double *result, unsigned int n)
{
  unsigned int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++ )
      result[i] += matrixa[i * n + j] * vectora[j];

}

/* Matricies are assumed to be square */
void matmatmult(double *matrixa, double *matrixb, double *result, unsigned int n)
{
  unsigned int i, j, k;
  for ( i=0; i<n; i++ )
    for( j=0; j<n; j++ )
      for( k=0; k<n; k++ )
	result[i * n + j] += matrixa[i * n + k] * matrixb[k * n + j];

}
