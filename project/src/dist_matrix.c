#include "dist_matrix.h"
#include "utils.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* Creates Distributed Block Cyclic Layout Matrix
  Requires:
   - p * q = size of communication group
   - m % p*bm == 0
   - n % q*bn == 0
*/
void dist_matrix_init(MPI_Comm comm, struct dbcl_t *mat,
		int p, int q,
		int bm, int bn,
		int m, int n,
		double (*f)(int i, int j)) {

  int rank, size;
  
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  assert(p*q == size);
  assert(m % p*bm == 0);
  assert(n % q*bn == 0);

  // M,N are the number of blocks
  int gM = m / bm;
  int gN = n / bn;
  
  int lM = gM / p;
  int lN = gN / q;

  // Initialize datastructure
  MPI_Comm_dup(comm, &(mat->comm));
  mat->p = p; mat->q = q;
  mat->lM = lM; mat->lN = lN;
  mat->bm = bm; mat->bn = bn;
  mat->m = m; mat->n = n;
  mat->value = malloc(sizeof(double) * (bm * bn) * (lM * lN));

  if (mat->value == NULL)
    exitWithError("dist_matrix_init: malloc failed ( mat->value )");
  
  // Fill matrix will values supplied from function f(i, j)
  int lbi, lbj;
  for ( lbj=0; lbj<lN; lbj++ ) {
    for ( lbi=0; lbi<lM; lbi++ ) {
      
      int block_offset = (lbi + lbj * lM) * (bm * bn);

      int i, j; 
      for (i=0; i<bm; i++ ) {
	for (j=0; j<bn; j++ ) {
	  
	  int gi = (rank % p) * bm + lbi * (p * bm) + i;
	  int gj = (rank / p) * bn + lbj * (q * bn) + j;

	  int offset = i + (j * bm);
	  if (gi < m && gj < n) {
	    mat->value[block_offset + offset] = f(gi, gj);
	  } else {
	    mat->value[block_offset + offset] = 0.0; 
	  }
	}
      }
    }
  }
}

void dist_matrix_free(struct dbcl_t *mat) {
  MPI_Comm_free(&(mat->comm));
  free(mat->value);
}

void dist_matrix_print(struct dbcl_t *mat) {
  int rank, size;

  MPI_Comm_rank(mat->comm, &rank);
  MPI_Comm_size(mat->comm, &size);

  double *global_mat;
  
  if (rank == ROOT) {
    global_mat = malloc(sizeof(double) * (mat->m * mat->n));
    if (global_mat == NULL)
      exitWithError("dist_matrix_print: malloc failed( global_mat )\n");
  }
  
  int send_size = (mat->lM * mat->lN) * (mat->bm * mat->bn);
  int recv_size = send_size;

  MPI_Gather(mat->value, send_size, MPI_DOUBLE, global_mat, recv_size, MPI_DOUBLE, ROOT, mat->comm);
  
  if (rank == ROOT) {
    printf("Print Matrix >>> (m, n): %d %d (lM, lN): %d %d (bm, bn): %d %d (p, q): %d %d\n", mat->m, mat->n, mat->lM, mat->lN, mat->bm, mat->bn, mat->p, mat->q);

    int colors[] = {31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96};
    int background[] = {003, 007};

    int gi, gj;
    for (gi=0; gi<mat->m; gi++ ) {
      for (gj=0; gj<mat->n; gj++ ) {
	int i_rank = (gi / mat->bm) % mat->p + ((gj / mat->bn) % mat->q) * mat->p;
	int proc_offset = i_rank * (mat->lM * mat->lN) * (mat->bm * mat->bn);

	int lbi = gi / (mat->bm * mat->p);
	int lbj = gj / (mat->bn * mat->q);

	int block_offset = (lbi + lbj * mat->lM) * (mat->bm * mat->bn);

	int i = gi % mat->bm;
	int j = gj % mat->bn;
	
	int index_offset = i + (j * mat->bm);

	char format_buffer[16];
	sprintf(format_buffer, "\033[%d;%dm", background[(lbi + lbj) % 2], colors[i_rank % 14]);
	printf("%s%2.1f\033[0m ", format_buffer, global_mat[proc_offset + block_offset + index_offset]);
      }
      printf("\n");
    }
    printf("\n");
    
    free(global_mat);
  }

  MPI_Barrier(mat->comm);
}

void dist_matrix_print_summary(struct dbcl_t *mat) {
  int rank;
  MPI_Comm_rank(mat->comm, &rank);

  printf("Print Local Matrix >>> (rank): %d (m, n): %d %d (lM, lN): %d %d (bm, bn): %d %d (p, q): %d %d\n", rank, mat->m, mat->n, mat->lM, mat->lN, mat->bm, mat->bn, mat->p, mat->q);
}

void dist_matrix_local_print(struct dbcl_t *mat) {
  int rank;
  MPI_Comm_rank(mat->comm, &rank);
  
  dist_matrix_print_summary(mat);

  int colors[] = {31, 32, 33, 34, 35, 36, 37, 90, 91, 92, 93, 94, 95, 96};
  int background[] = {003, 007};
  
  int lbi, lbj;
  int i, j;
  for ( lbi=0; lbi<mat->lM; lbi++ ) {
    for ( i=0; i<mat->bm; i++ ) {
      for (lbj=0; lbj<mat->lN; lbj++ ) {
	for (j=0; j<mat->bn; j++) {
	  int block_offset = (lbi + lbj * mat->lM) * (mat->bm * mat->bn);
	  int index_offset = i + j * mat->bm;

	  char format_buffer[16];
	  sprintf(format_buffer, "\033[%d;%dm", background[(lbi + lbj) % 2], colors[rank % 14]);
	  printf("%s%2.0f\033[0m ", format_buffer, mat->value[block_offset + index_offset]);
	}
      }
      printf("\n");
    }
  }
  printf("\n");
}

/* Copies idential copy of matrix a to matrix b distributed.
 Assumes that b is uninitialized*/
void dist_matrix_deepcopy(struct dbcl_t *a, struct dbcl_t *b) {
  MPI_Comm_dup(a->comm, &(b->comm));

  b->p = a->p; b->q = a->q;
  b->lM = a->lM; b->lN = a->lN;
  b->bm = a->bm; b->bn = a->bn;
  b->m = a->m; b->n = a->n;

  b->value = malloc(sizeof(double) * (b->lM * b->lN) * (b->bm * b->bn));
  memcpy(b->value, a->value, sizeof(double) * (a->bm * a->bn) * (a->lM * a->lN));
}

/* Copies rectangular blocks of distributed matrix a to distributed matrix b
   Requires:
   Both matrix structs are preinitialized.
   Requested block to copy must fit in both matricies
 */
void dist_matrix_block_copy(struct dbcl_t *a, int aibm, int aibn,
			    struct dbcl_t *b, int bibm, int bibn,
			    int M, int N)
{
#ifdef DEBUG_MATRIX
  int rank, size;
  MPI_Comm_rank(a->comm, &rank);
  MPI_Comm_size(a->comm, &size);

  if (rank == ROOT) {
    printf("dist_matrix_block_copy:\n");
    printf("Parameters: {aibm: %d, aibn: %d, bibm: %d, bibn: %d, M: %d, N: %d}\n", aibm, aibn, bibm, bibn, M, N);
    printf("Matrix A\n");
    dist_matrix_local_print(a);
    printf("Matrix B\n");
    dist_matrix_local_print(b);
  }
#endif

  // Ensure requested block fits in all matricies
  assert(aibm >= 0); assert(aibn >= 0);
  assert(a->lM >= aibm + M); assert(a->lN >= aibn + N);

  assert(bibm >= 0); assert(bibn >= 0);
  assert(b->lM >= bibm + M); assert(b->lN >= bibn + N);

  // Ensure block sizes are equal
  assert(a->bm == b->bm); assert(a->bn == b->bn);
  
  int i;
  for (i=0; i<N; i++ ) {
    int a_src_offset = (aibm + (aibn + i) * a->lM) * (a->bm * a->bn);
    int b_dest_offset = (bibm + (bibn + i) * b->lM) * (b->bm * b->bn);
    int b_buffer_size = (M) * (b->bm * b->bn) * sizeof(double);
    memcpy(b->value + b_dest_offset, a->value + a_src_offset, b_buffer_size);
  }
}


/* Does requested opperation on distributed block of memory
   - opp
       BLOCK_SUB C = A - B
       BLOCK_ADD C = A + B

   you can supply identical matricies for different arguments
   (e.g. dist_matric_opp(a, 0, 0, a, 0, 0, a, 0, 0, a->lM, a->lN, BLOCK_SUB) )
 */
void dist_matrix_block_opp(struct dbcl_t *a, int aibm, int aibn,
			   struct dbcl_t *b, int bibm, int bibn,
			   struct dbcl_t *c, int cibm, int cibn,
			   int M, int N, enum block_opp opp)
{
#ifdef DEBUG_MATRIX
  int rank, size;
  MPI_Comm_rank(a->comm, &rank);
  MPI_Comm_size(a->comm, &size);

  if (rank == ROOT) {
    printf("dist_matrix_block_copy:\n");
    printf("Parameters: {aibm: %d, aibn: %d, bibm: %d, bibn: %d, cibm: %d, cibn: %d, M: %d, N: %d, opp: %s}\n", aibm, aibn, bibm, bibn, M, N, (opp == BLOCK_ADD) ? "add" : "sub");
    printf("Matrix A\n");
    dist_matrix_local_print(a);
    printf("Matrix B\n");
    dist_matrix_local_print(b);
    printf("Matrix C\n");
    dist_matrix_local_print(c);
  }
#endif

  // Ensure requested block fits in all matricies
  assert(aibm >= 0); assert(aibn >= 0);
  assert(a->lM >= aibm + M); assert(a->lN >= aibn + N);

  assert(bibm >= 0); assert(bibn >= 0);
  assert(b->lM >= bibm + M); assert(b->lN >= bibn + N);

  assert(cibm >= 0); assert(cibn >= 0);
  assert(c->lM >= cibm + M); assert(c->lN >= cibn + N);

  // Ensure block sizes are equal
  assert(a->bm == b->bm); assert(b->bm == c->bm);
  assert(a->bn == b->bn); assert(b->bn == c->bn);
  
  
  int i, j;
  int bi, bj;
  
  switch (opp) {
  case(BLOCK_ADD):
    for (bi=0; bi<M; bi++ ) {
      for (bj=0; bj<N; bj++ ) {
	for (i=0; i<a->bm; i++ ) {
	  for (j=0; j<a->bn; j++ ) {
	    int a_offset = (aibm + bi + (aibn + bj) * a->lM) * (a->bm * a->bn) + i + j * a->bm;
	    int b_offset = (bibm + bi + (bibn + bj) * b->lM) * (b->bm * b->bn) + i + j * b->bm;
	    int c_offset = (cibm + bi + (cibn + bj) * c->lM) * (c->bm * c->bn) + i + j * c->bm;

	    c->value[c_offset] = a->value[a_offset] + b->value[b_offset] ;
	  }
	}
      }
    }
    break;
  case(BLOCK_SUB):
    for (bi=0; bi<M; bi++ ) {
      for (bj=0; bj<N; bj++ ) {
	for (i=0; i<a->bm; i++ ) {
	  for (j=0; j<a->bn; j++ ) {
	    int a_offset = (aibm + bi + (aibn + bj) * a->lM) * (a->bm * a->bn) + i + j * a->bm;
	    int b_offset = (bibm + bi + (bibn + bj) * b->lM) * (b->bm * b->bn) + i + j * b->bm;
	    int c_offset = (cibm + bi + (cibn + bj) * c->lM) * (c->bm * c->bn) + i + j * c->bm;

	    c->value[c_offset] = a->value[a_offset] - b->value[b_offset];
	  }
	}
      }
    }
    break;
  }
}

void dist_matrix_to_block(struct dbcl_t *a) {
  
  double *temp_buffer = malloc(sizeof(double) * (a->lM * a->lN) * (a->bm * a->bn));

  int bi, bj;
  int j;
  for (bi=0; bi<a->lM; bi++)
    for (bj=0; bj<a->lN; bj++)
      for (j=0; j<a->bn; j++) {
	memcpy(temp_buffer + bi * a->bm + (bj * a->lM) * (a->bm * a->bn) + j * a->bm * a->lM,
	       a->value + (bi + bj * a->lM) * (a->bm * a->bn) + j * a->bm, sizeof(double) * a->bm);
	  }
  
  double *swap_ptr;
  swap_ptr = a->value;
  a->value = temp_buffer;
  temp_buffer = swap_ptr;

  a->bm *= a->lM; a->bn *= a->lN;
  a->lM = 1; a->lN = 1;
  
  free(temp_buffer);
}

void block_to_dist_matrix(struct dbcl_t *a, int bm, int bn) {
  
  double *temp_buffer = malloc(sizeof(double) * (a->lM * a->lN) * (a->bm * a->bn));

  int lM = a->bm/bm; int lN = a->bn/bn;
  int bi, bj;
  int j;
  for (bi=0; bi<lM; bi++)
    for (bj=0; bj<lN; bj++)
      for (j=0; j<bn; j++) {
	memcpy(temp_buffer + (bi + bj * lM) * (bm * bn) + j * bm,
	       a->value + bi * bm + (bj * lM) * (bm * bn) + j * bm * lM, sizeof(double) * bm);
	  }
  
  double *swap_ptr;
  swap_ptr = a->value;
  a->value = temp_buffer;
  temp_buffer = swap_ptr;

  a->bm = bm; a->bn = bn;
  a->lM = lM; a->lN = lN;
  
  free(temp_buffer);
}

