#ifndef DIST_MATRIX_H
#define DIST_MATRIX_H

#include "mpi.h"

/* Distributed Block Cyclic Layout
 comm - communication group used to store matrix
 lM, lN - dim of local blocks (on procesor) 
 gM, gN - dim of global blocks (on all processors)
 m, n - dim of global matrix
 value - stores local values of matrix in column major format*/
struct dbcl_t {
  MPI_Comm comm;
  int p, q;
  int lM, lN;
  int bm, bn;
  int m, n;
  double *value;
};

enum block_opp {
  BLOCK_ADD,
  BLOCK_SUB
};

void dist_matrix_init(MPI_Comm comm, struct dbcl_t *mat,
		int p, int q,
		int bm, int bn,
		int m, int n,
		double (*f)(int i, int j));

void dist_matrix_deepcopy(struct dbcl_t *a, struct dbcl_t *b);
void dist_matrix_block_copy(struct dbcl_t *a, int aibm, int aibn,
			    struct dbcl_t *b, int bibm, int bibn,
			    int M, int N);
void dist_matrix_free(struct dbcl_t *mat);

void dist_matrix_block_opp(struct dbcl_t *a, int aibm, int aibn,
			   struct dbcl_t *b, int bibm, int bibn,
			   struct dbcl_t *c, int cibm, int cibn,
			   int M, int N, enum block_opp opp);

void dist_matrix_print(struct dbcl_t *mat);
void dist_matrix_print_summary(struct dbcl_t *mat);
void dist_matrix_local_print(struct dbcl_t *mat);

void dist_matrix_to_block(struct dbcl_t *a);
void block_to_dist_matrix(struct dbcl_t *a, int bm, int bn);

#endif
