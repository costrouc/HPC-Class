#ifndef GEMM_OPP_H
#define GEMM_OPP_H

void dgemm_1( int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc );
void dgemm_2( int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc );
void dgemm_3( int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc );
void dgemm_4( int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc );

#endif
