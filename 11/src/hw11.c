#include "hw11.h"

#include <mpi.h>
#include <cblas.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define red   "\033[0;31m"
#define cyan  "\033[0;36m"
#define green "\033[0;32m"
#define blue  "\033[0;34m"
#define brown  "\033[0;33m"
#define purple "\033[0;35m"
#define lightblue "\033[1;34m"
#define lightgreen "\033[1;32m"
#define lightcyan "\033[1;36m"
#define lightred "\033[1;31m"
#define lightpurple "\033[1;35m"
#define none   "\033[0m"

int mpi_error(const int error_code) {
  char string[256];
  int length;

  MPI_Error_string(error_code, string, &length);
  printf("Error: %s\n", string);
  
  exit(1);
}

/* m - number of rows of processors
   n - number of columns of processors
   m*n = total number of processors
*/
int CreateGemmCommGroups(const int m, const int n, const MPI_Comm comm, MPI_Comm *comm_row, MPI_Comm *comm_col) {
  int retval;
  int rank, size;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  assert(size == m*n);
  
  // Create row communication group
  if ((retval = MPI_Comm_split(comm, rank % m, rank / m, comm_row)) != MPI_SUCCESS)
    mpi_error(retval);
  
  // Create column communication group
  if ((retval = MPI_Comm_split(comm, rank / m, rank % m, comm_col)) != MPI_SUCCESS)
    mpi_error(retval);

  return 0;
}

  
int PassTokenComm(const MPI_Comm comm, int *send_message, int *recv_message ) {
  int rank, size;
  
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  int tag = 0;
  
  MPI_Status status;
  int retval;
  
  int recv_neigh = (rank - 1 < 0) ? size - 1 : rank - 1;
  int send_neigh = (rank + 1) % size;
  
  if (rank == 0) {

    if ((retval = MPI_Send(send_message, 1, MPI_INT, send_neigh, tag, comm)) != MPI_SUCCESS)
      mpi_error(retval);

    if ((retval = MPI_Recv(recv_message, 1, MPI_INT, recv_neigh, tag, comm, &status)) != MPI_SUCCESS)
      mpi_error(retval);

  } else {

    if ((retval = MPI_Recv(recv_message, 1, MPI_INT, recv_neigh, tag, comm, &status)) != MPI_SUCCESS)
      mpi_error(retval);
    
    if ((retval = MPI_Send(send_message, 1, MPI_INT, send_neigh, tag, comm)) != MPI_SUCCESS)
      mpi_error(retval);
  }

  //printf("Node %d recieved message from Node %d: %d\n", rank, recv_neigh, recieved_message);
  
  return 0;
}

int CreateMatricies(MPI_Comm comm,
		    int p, int q,
		    int bm, int bn, int bk,
		    int gm, int gn, int gk,
		    double *a, double *b, double *c,
		    double (*fa)(int i, int j),
		    double (*fb)(int i, int j),
		    double (*fc)(int i, int j)) {
  
  int rank, size;
  
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  // Check that parameters form valid configuration
  assert(p*q == size);
  assert(gm % bm == 0);
  assert(gk % bk == 0);
  assert(gn % bn == 0);
  assert(bm % p == 0);
  assert(bn % q == 0);
  assert(bk % p == 0);
  assert(bk % q == 0);
  assert(bk / p == bk / q);

  int
    pm = bm / p,
    pn = bn / q,
    pk = bk / q;

  int
    m = gm / bm,
    n = gn / bn,
    k = gk / bk;

  int i, j, l;
  int pi, pj, pl;
  int bi, bj, bl;
  int gi, gj, gl;
  
  // Form A matrix
  for ( bi = 0; bi < m; bi++)
    for ( bl = 0; bl < k; bl++ ) {
      // Calculate processor location by column
      pi = rank % q;
      pl = rank / p;
      
      for ( i = 0; i < pm; i++ )
	for ( l = 0; l < pk; l++ )
	  {
	    gi = (bi * bm) + (pi * pm) + i;
	    gl = (bl * bk) + (pl * pk) + l;

	    a[(bi + bl * m) * (pm * pk) + (i + pm * l)] = fa(gi, gl);
	  }
    }

  // Form B matrix
  for ( bl = 0; bl < k; bl++)
    for ( bj = 0; bj < n; bj++ ) {
      // Calculate processor location by column
      pl = rank % q;
      pj = rank / p;
      
      for ( l = 0; l < pk; l++ )
	for ( j = 0; j < pn; j++ )
	  {
	    gl = (bl * bk) + (pl * pk) + l;
	    gj = (bj * bn) + (pj * pn) + j;

	    b[(bl + bj * k) * (pk * pn) + (l + pk * j)] = fb(gl, gj);
	  }
    }

  // Form C matrix
  for ( bi = 0; bi < m; bi++)
    for ( bj = 0; bj < n; bj++ ) {
      // Calculate processor location by column
      pi = rank % q;
      pj = rank / p;
      
      for ( i = 0; i < pm; i++ )
	for ( j = 0; j < pn; j++ )
	  {
	    gi = (bi * bm) + (pi * pm) + i;
	    gj = (bj * bn) + (pj * pn) + j;

	    c[(bi + bj * m) * (pm * pn) + (i + pm * j)] = fc(gi, gj);
	  }
    }

  return 0;
}

int ShiftMPIMatrixLeft(MPI_Comm comm_row, int gm, int gn, int bm, int bn, int p, int q, double *a) {
  int rank, size;

  MPI_Comm_rank (comm_row, &rank);
  MPI_Comm_size (comm_row, &size);

  int tag = 0;
  
  MPI_Status status;
  int retval;

  int recv_neigh = (rank + 1) % size;
  int send_neigh = (rank - 1 < 0) ? size - 1 : rank - 1;

  double *a_buffer = malloc(sizeof(double) * (gm*gn) / (p*q));
  if (a_buffer == NULL) {
    fprintf(stderr, "Failed to malloc memory for arrays\n");
    exit(1);
  }

  int
    pm = bm / p,
    pn = bn / q;

  int
    m = gm / bm,
    n = gn / bn;

  int a_offset;
  int a_buffer_offset;
  int message_size = (pm*pn);
  
  int bi, bj;
    
  if (rank == 0) {
    for ( bi=0; bi<m; bi++ )
      for ( bj=0; bj<n; bj++ ) {
	a_offset = (bi + bj * m) * (pm*pn);
	
	if ((retval = MPI_Send(a + a_offset, message_size, MPI_DOUBLE, send_neigh, tag, comm_row)) != MPI_SUCCESS)
	  mpi_error(retval);
      }

    for ( bi=0; bi<m; bi++ )
      for ( bj=0; bj<n; bj++ ) {
	
	a_buffer_offset = (bi + bj * m) * (pm*pn);
	
	if ((retval = MPI_Recv(a_buffer + a_buffer_offset, message_size, MPI_DOUBLE, recv_neigh, tag, comm_row, &status)) != MPI_SUCCESS)
	  mpi_error(retval);
      }

  } else {
    for ( bi=0; bi<m; bi++ )
      for ( bj=0; bj<n; bj++ ) {
	if (recv_neigh == 0) {
	  a_buffer_offset = (bi + (bj == 0 ? n-1 : bj-1) * m) * (pm*pn);
	} else {
	  a_buffer_offset = (bi + bj * m) * (pm*pn);
	}
	
	if ((retval = MPI_Recv(a_buffer + a_buffer_offset, message_size, MPI_DOUBLE, recv_neigh, tag, comm_row, &status)) != MPI_SUCCESS)
	  mpi_error(retval);
      }
    
    for ( bi=0; bi<m; bi++ )
      for ( bj=0; bj<n; bj++ ) {
	a_offset = (bi + bj * m) * (pm*pn);
	
	if ((retval = MPI_Send(a + a_offset, message_size, MPI_DOUBLE, send_neigh, tag, comm_row)) != MPI_SUCCESS)
	  mpi_error(retval);
      }
  }

  // Copy data from buffer to Matrix
  int i;
  for ( i=0; i<(gm*gn) / (p*q); i++ )
    a[i] = a_buffer[i];
    
  free(a_buffer);

  return 0;
}

void PrintMPIMatrix(MPI_Comm comm, int gm, int gn, int bm, int bn, int p, int q, double *a) {
  int rank, size;
  int root = 0;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  double *A = malloc(sizeof(double) * gm * gn);
  if (A == NULL) {
    fprintf(stderr, "Failed to malloc memory for arrays\n");
    exit(1);
  }

  
  int
    pm = bm / p,
    pn = bn / q;
  
  int
    m = gm / bm,
    n = gn / bn;
  
  MPI_Gather(a, (gm*gn) / (p*q), MPI_DOUBLE, A, (gm*gn) / (p*q) , MPI_DOUBLE, root, comm);

  int i,j,l;
  int bi, bj;
  int pi, pj;

  if (rank == root) {

    char *colors[] = {red, cyan, green, blue, brown, purple,
		      lightblue, lightgreen, lightcyan, lightred, lightpurple};

   
    
    printf("Matrix: (%d %d) Block: (%d %d) Processor (%d %d)\n", gm, gn, bm, bn, p, q);
    
    for ( bi=0; bi<m; bi++ ) {
      for (i=0; i<bm; i++ ) {
	pi = i / (bm/p);

	for ( bj=0; bj<n; bj++ ) {
	  for (pj=0; pj<q; pj++ ) {

	    int curr_rank = pi + pj*p;

	    for (j=0; j<pn; j++) {
	      int proc_offset = curr_rank * (gm*gn) / (p*q);
	      int block_offset = (bi + bj*m) * (pm*pn);
	      
	      double value = A[proc_offset + block_offset + i % (bm/p) + j*pm];
	      //double value = pi * 10.0 + pj;
	      if( value < 0.0000001 && value > -0.0000001) {
		printf("      ");
	      } else {
		printf("%s%5.4f%s ", colors[curr_rank % 11], value, none);
	      }
	    }
	  }
	}
	printf("\n");
      }
    }

    printf("\n");

    free(A);
  }
}
/*
void PrintMPIMatrix(MPI_Comm comm, int gm, int gn, int bm, int bn, int p, int q, double *a) {
  int rank, size;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  int
    pm = bm / p,
    pn = bn / q;
  
  int
    m = gm / bm,
    n = gn / bn;

  int bi, bj;
  int pi, pj;
  int gi, gj;
  int i, j, l;

  for ( bi = 0; bi < m; bi++)
    for ( bj = 0; bj < n; bj++ ) {

      if (rank == 0)
	printf("Block (%d %d)\n", bi, bj);
      
      for (l = 0; l<size; l++) {
	MPI_Barrier(comm);

	if (l == rank) {
	  pi = rank % q;
	  pj = rank / p;

	  printf("Rank %d: (%d %d)\n", rank, pi, pj);
	  for ( i = 0; i < pm; i++ ) {
	    for ( j = 0; j < pn; j++ ) {
	      gi = (bi * bm) + (pi * pm) + i;
	      gj = (bj * bn) + (pj * pn) + j;

	      printf("(%d %d ) %2.2f ", gi , gj, a[(bi + bj * m) * (pm * pn) + i + j * pm]);
	    }
	    printf("\n");

	  }
	  printf("\n");
	}
      }
    }
}
*/

int pdgemm(MPI_Comm comm, int p, int q, int bm, int bn, int bk, int gm, int gn, int gk, double *a, double *b, double *c) {

  int rank, size;
  
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  MPI_Comm comm_row, comm_col;
  CreateGemmCommGroups(p, q, comm, &comm_row, &comm_col);

  int col_rank;
  MPI_Comm_rank (comm_col, &col_rank);
  
  int
    pm = bm / p,
    pn = bn / q,
    pk = bk / q;

  int
    m = gm / bm,
    n = gn / bn,
    k = gk / bk;


  int bi, bj, bl;

  double *b_buffer = malloc(sizeof(double) * pk * pn * n);
  if (b_buffer == NULL) {
    fprintf(stderr, "Failed to malloc memory for arrays\n");
    exit(1);
  }
  
  int i, j, l;
  for ( l=0; l<p*k; l++ ) {

    for (i=0; i<n; i++ ) {
      int buffer_offset = (i*pm*pn);

      int col_J = i*q + rank/p;
      int row_I = (col_J + l) % (p*k);

      int bcast_origin_rank = row_I % p;

      if (col_rank == bcast_origin_rank) {
	bl = row_I / p;
	bj = col_J / q;

	for ( j=0; j<pk*pn; j++ )
	  b_buffer[buffer_offset + j] = b[(bl + bj * k) * (pk * pn) + j];

      }
      //printf("(%d %d) [%d %d] %d ", row_I, col_J, bl, bj, bcast_origin_rank);
      MPI_Bcast(b_buffer + buffer_offset, pk*pn, MPI_DOUBLE, bcast_origin_rank, comm_col);

    }

    /*
      printf("Rank %d ", rank);
      for ( i=0; i< pk*pn*n; i++ ) {
      printf("%2.2f ", b_buffer[i]);
      }
      printf("\n");
    */
    
    for ( bi=0; bi<m; bi++ ) {
      for ( bj=0; bj<n; bj++ ) {

	int a_offset = (bi + bj * m) * (pm * pk);
	int b_offset = bj * pk * pn;
	int c_offset = (bi + bj * m) * (pm * pn);
	
	cblas_dgemm (CblasColMajor, CblasNoTrans , CblasNoTrans, pm, pn, pk, 1.0, a + a_offset, pm, b_buffer + b_offset, pk, 1.0, c + c_offset, pm); 
      }
    }

    ShiftMPIMatrixLeft(comm_row, gm, gn, bm, bn, p, q, a);
  }

  free(b_buffer);

  return 0;
}
