#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int dge_cholesky_qr(long int m, long int n, double *A, double *Q, double *R)
{
  /* All opperations are done row major */
  
  double cn = 200.0;

  double *G = (double *) malloc(sizeof(double) * n * n);

  double *u = (double *) malloc(sizeof(double) * n * n);
  double *s = (double *) malloc(sizeof(double) * n);
  double *vt = (double *) malloc(sizeof(double) * n * n);

  double *r = (double *) malloc(sizeof(double) * n * n);
    
  double *temp = (double *) malloc(sizeof(double) * n * n); // temporary calculations
  double *devPtrQ, *devPtrG, *devPtrr; 
  
  int iter = 0;
  int max_iter = 100;

  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;

  /* Allocate memory to store devPtrQ[m, n] */
  cudaStat = cudaMalloc ((void**)&devPtrQ, m*n*sizeof(double));
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed for devPtrQ");
    return EXIT_FAILURE;
  }

  /* Allocate memory to store devPtrG[n, n] */
  cudaStat = cudaMalloc ((void**)&devPtrG, n*n*sizeof(double));
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed for devPtrG");
    return EXIT_FAILURE;
  }

  /* Allocate memory to store devPtrr[n, n] */
  cudaStat = cudaMalloc ((void**)&devPtrr, n*n*sizeof(double));
  if (cudaStat != cudaSuccess) {
    printf ("device memory allocation failed for devPtrG");
    return EXIT_FAILURE;
  }
  
  /* Initialize cublas */
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }

  /* Q[m,n] = A[m,n] */
  /* Copy Q[m, n] to devPtrQ[m, n] */
  stat = cublasSetMatrix (m, n, sizeof(double), A, m, devPtrQ, m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data download failed");
    cudaFree (devPtrQ);
    cudaFree (devPtrG);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }

  double alpha = 1.0;
  double beta = 0.0;
  /* devPtrG[n,n] =  devPtrQ[m,n]^T * devPtrQ[m,n] */
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
	      n, n, m, &alpha, devPtrQ, m, devPtrQ, m, &beta, devPtrG, n);

  /* Copy devPtrG[n, n] to G[n, n] */
  stat = cublasGetMatrix (n, n, sizeof(double), devPtrG, n, G, n);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data upload failed");
    cudaFree (devPtrQ);
    cudaFree (devPtrG);
    cublasDestroy(handle);        
    return EXIT_FAILURE;
  }

  long int i, j;
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

      /* Copy invr[n, n] to devPtrr[n, n] */
      stat = cublasSetMatrix (n, n, sizeof(double), r, n, devPtrr, n);
      if (stat != CUBLAS_STATUS_SUCCESS) {
	printf ("data download failed");
	cudaFree (devPtrQ);
	cudaFree (devPtrG);
	cublasDestroy(handle);
	return EXIT_FAILURE;
      }
      
      /* devPtrQ[m, n] = devPtrQ[m, n] * devPtrr[n, n] */
      double alpha = 1.0;
      cublasDtrmm( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, devPtrr, n, devPtrQ, m, devPtrQ, m);
      
      if ( cn > 100.0 )
	{
	  double alpha = 1.0;
	  double beta = 0.0;
	  /* devPtrG[n,n] =  devPtrQ[m,n]^T * devPtrQ[m,n] */
	  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
		      n, n, m, &alpha, devPtrQ, m, devPtrQ, m, &beta, devPtrG, n);

	  /* Copy devPtrG[n, n] to G[n, n] */
	  stat = cublasGetMatrix (n, n, sizeof(double), devPtrG, n, G, n);
	  if (stat != CUBLAS_STATUS_SUCCESS) {
	    printf ("data upload failed");
	    cudaFree (devPtrQ);
	    cudaFree (devPtrG);
	    cublasDestroy(handle);        
	    return EXIT_FAILURE;
	  }
	}

      if (iter > max_iter)
	{
	  fprintf(stderr, "Max iterations reached %d\n", iter);
	  return;
	}
    }

  /* Copy devPtrQ[n, n] to Q[n, n] */
  stat = cublasGetMatrix (m, n, sizeof(double), devPtrQ, m, Q, m);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data upload failed");
    cudaFree (devPtrQ);
    cublasDestroy(handle);        
    return EXIT_FAILURE;
  }

  
  free(G); free(u); free(s); free(vt); free(r); free(temp);

  cudaFree(devPtrQ); cudaFree(devPtrG);
  cublasDestroy(handle);
  
  return EXIT_SUCCESS;
}
