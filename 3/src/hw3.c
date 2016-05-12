#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <stdio.h>

void dge_cholesky_qr(long int m, long int n, double *A, double *Q, double *R)
{
  /* All opperations are done row major */
  
  double cn = 200.0;

  double *G = (double *) malloc(sizeof(double) * n * n);

  double *u = (double *) malloc(sizeof(double) * n * n);
  double *s = (double *) malloc(sizeof(double) * n);
  double *vt = (double *) malloc(sizeof(double) * n * n);

  double *r = (double *) malloc(sizeof(double) * n * n);
    
  double *temp = (double *) malloc(sizeof(double) * n * n); // temporary calculations
  int iter = 0;
  int max_iter = 100;
      
  /* Q[m,n] = A[m,n] */
  long int i, j;
  for ( i=0; i<m*n; i++ )
    Q[i] = A[i];

  /* G[n,n] =  Q[m,n]^T * Q[m,n] */
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
	      n, n, m, 1.0, Q, m, Q, m, 0.0, G, n);
  
  /* R[n,n] = Identity[n,n] */
  for ( j=0; j<n; j++ )
    for ( i=0; i<n; i++ )
      {
	if (i == j)
	  R[j * n + i] = 1.0;
	else
	  R[j * n + i] = 0.0;
      }

  while ( cn > 100.0 )
    {
      iter++;
      
      /* u[n,n], s[n,n], vt[n,n] = svd(G[n,n]) */
      LAPACKE_dgesvd( LAPACK_COL_MAJOR , 'N', 'A', n, n, G, n, s, u, n, vt, n, temp );
      
      /* r[n, n] =  sqrt(s) * vt[n, n] */
      for ( j=0; j<n; j++ )
	for ( i=0; i<n; i++ )
	  r[j * n + i] = sqrt(s[i]) * vt[j * n + i];

      /* r[n, n] = qr(r[n, n])  */
      LAPACKE_dgeqrf( LAPACK_COL_MAJOR , n, n, r, n, temp );

      /* R[n, n] = r[n, n] * R[n, n] */
      cblas_dtrmm( CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, r, n, R, n);

      cn = sqrt(s[0] / s[n-1]);

      /* r[n, n]  = inv(r[n, n]) */
      LAPACKE_dtrtri( LAPACK_COL_MAJOR, 'U', 'N', n, r, n);

      /* Q[m, n] = Q[m, n] * r[n, n] */
      cblas_dtrmm( CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1.0, r, n, Q, m);
      
      if ( cn > 100.0 )
	{
	  /* G[n,n] =  Q[m,n]^T * Q[m,n] */
	  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		      n, n, m, 1.0, Q, m, Q, m, 0.0, G, n);
	}

      if (iter > max_iter)
	{
	  fprintf(stderr, "Max iterations reached %d\n", iter);
	  return;
	}
    }

  free(G); free(u); free(s); free(vt); free(r); free(temp);
}
