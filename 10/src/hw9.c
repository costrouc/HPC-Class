// Column major definition (inspiration from cblas source)
#define A(I,J) a[(I) + (J) * lda]
#define B(I,J) b[(I) + (J) * ldb]
#define C(I,J) c[(I) + (J) * ldc]

void ijk (int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int i, j, l;
  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
      for ( l=0; l<k; l++ )
	C(i,j) = C(i,j) + A(i,l) * B(l,j);
}

void jik (int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int i, j, l;
  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      for ( l=0; l<k; l++ )
	C(i,j) = C(i,j) + A(i,l) * B(l,j);
}

void ikj (int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int i, j, l;
  for ( i=0; i<m; i++ )
    for ( l=0; l<k; l++ )
      for ( j=0; j<n; j++ )
	C(i,j) = C(i,j) + A(i,l) * B(l,j);
}

void jki (int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int i, j, l;
  for ( j=0; j<n; j++ )
    for ( l=0; l<k; l++ )
      for ( i=0; i<m; i++ )
	C(i,j) = C(i,j) + A(i,l) * B(l,j);
}

void kji (int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int i, j, l;
  for ( l=0; l<k; l++ )
    for ( i=0; i<m; i++ )
      for ( j=0; j<n; j++ )
	C(i,j) = C(i,j) + A(i,l) * B(l,j);
}

void kij (int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int i, j, l;
  for ( l=0; l<k; l++ )
    for ( i=0; i<m; i++ )
      for ( j=0; j<n; j++ )
	C(i,j) = C(i,j) + A(i,l) * B(l,j);
}

