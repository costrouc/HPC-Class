/* caps.c
 Written by: Chris Ostrouchov
 This algorithms is an parallel algorithm for doing fast matrix multiplication. The algorithm originates from the paper Grey Ballard, James Demmel, Olga Holtz, Benjamin Lipshitz, and Oded Schwartz, "Communication-Optimal Parallel Algorithm for Strassen's Matrix Multiplication". It is currently the fastest algorithm for computing the matrix matrix multiplication. To achieve this speedup they use Strassen's Algorithm. This algorithm has many benefits over the classical matrix multiplication (as well as a few problems, mainly stability). */

#include "dist_matrix.h"
#include "utils.h"

#if defined __INTEL_COMPILER
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>

void _caps_create_matricies_bfs(struct dbcl_t *a, struct dbcl_t *b, struct dbcl_t *c,
				struct dbcl_t *t, struct dbcl_t *s, struct dbcl_t *q,
				int k)
{
  /*
    if step == 1 (row wise join)
      T = [ T_0 | T_1 | T_2 | T_3 | T_4 | T_5 | T_6 ]
      S = [ S_0 | S_1 | S_2 | S_3 | S_4 | S_5 | S_6 ]
      Q = [ Q_0 | Q_1 | Q_2 | Q_3 | Q_4 | Q_5 | Q_6 ]
    */
  
  // Initilize the datastructures for S, T, Q
  t->lM = a->lM/2; t->lN = a->lN/2*7;
  t->bm = a->bm; t->bn = a->bn;
  t->m = a->m/2; t->n = a-> n/2;

  s->lM = b->lM/2; s->lN = b->lN/2*7;
  s->bm = b->bm; s->bn = b->bn;
  s->m = b->m/2; s->n = b-> n/2;

  q->lM = c->lM/2; q->lN = c->lN/2;
  q->bm = c->bm; q->bn = c->bn;
  q->m = c->m/2; q->n = c-> n/2;

  int t_iM = a->lM/2, t_iN = a->lN/2;
  int s_iM = b->lM/2, s_iN = b->lN/2;
  int t_offsetN = t_iN, s_offsetN = s_iN;

  int rank, size;
  MPI_Comm_rank(a->comm, &rank);
  MPI_Comm_size(a->comm, &size);
  
  if(k % 2==0) { // Row
    MPI_Comm_split(a->comm, (rank / a->p) % 7, ((rank % a->p) * a->q + rank / a->p) / 7, &(t->comm));
    MPI_Comm_split(b->comm, (rank / a->p) % 7, ((rank % b->p) * b->q + rank / b->p) / 7, &(s->comm));
    MPI_Comm_split(c->comm, (rank / a->p) % 7, ((rank % c->p) * c->q + rank / c->p) / 7, &(q->comm));
    
    t->p = a->p; t->q = a->q/7;
    s->p = b->p; s->q = b->q/7;
    q->p = c->p; q->q = c->q/7;

    q->lN = q->lN * 7;
  } else { // Column
    MPI_Comm_split(a->comm, rank % 7, rank / 7, &(t->comm));
    MPI_Comm_split(b->comm, rank % 7, rank / 7, &(s->comm));
    MPI_Comm_split(c->comm, rank % 7, rank / 7, &(q->comm));
    
    t->p = a->p/7; t->q = a->q;
    s->p = b->p/7; s->q = b->q;
    q->p = c->p/7; q->q = c->q;

    q->lM = q->lM * 7;
  }

  t->value = malloc(sizeof(double) * (t->lM * t->lN) * (t->bm * t->bn));
  if (t->value == NULL)
    exitWithError("_caps_create_matricies_bfs: malloc failed ( t->value )");

  s->value = malloc(sizeof(double) * (s->lM * s->lN) * (s->bm * s->bn));
  if (s->value == NULL)
    exitWithError("_caps_create_matricies_bfs: malloc failed ( s->value )");

  q->value = calloc((q->lM * q->lN) * (q->bm * q->bn), sizeof(double));
  if (q->value == NULL)
    exitWithError("_caps_create_matricies_bfs: calloc failed ( q->value )");
  
  int i;
  // T_0 = A_11          S_0 = B_11
  i = 0;
  dist_matrix_block_copy(a, 0, 0, t, 0, t_offsetN * i, t_iM, t_iN);
  dist_matrix_block_copy(b, 0, 0, s, 0, s_offsetN * i, s_iM, t_iN);
  // T_1 = A_12          S_1 = B_21
  i = 1;
  dist_matrix_block_copy(a, 0, a->lN/2, t, 0, t_offsetN * i, t_iM, t_iN);
  dist_matrix_block_copy(b, b->lM/2, 0, s, 0, s_offsetN * i, s_iM, s_iN);
  //T_2 = A_21 + A_22    S_2 = B_12 - B_11
  i = 2;
  dist_matrix_block_opp(a, a->lM/2, 0, a, a->lM/2, a->lN/2, t, 0, t_offsetN * i, t_iM, t_iN, BLOCK_ADD);
  dist_matrix_block_opp(b, 0, b->lN/2, b, 0, 0, s, 0, s_offsetN * i, s_iM, s_iN, BLOCK_SUB);
  //T_3 = T_2 - A_11     S_3 = B_22 - S_2
  i = 3;
  dist_matrix_block_opp(t, 0, t_offsetN * 2, a, 0, 0, t, 0, t_offsetN * i, t_iM, t_iN, BLOCK_SUB);
  dist_matrix_block_opp(b, b->lM/2, b->lN/2, s, 0, s_offsetN * 2, s, 0, s_offsetN * i, s_iM, s_iN, BLOCK_SUB);
  //T_4 = A_11 - A_21    S_4 = B_22 - B_12
  i = 4;
  dist_matrix_block_opp(a, 0, 0, a, a->lM/2, 0, t, 0, t_offsetN * i, t_iM, t_iN, BLOCK_SUB);
  dist_matrix_block_opp(b, b->lM/2, b->lN/2, b, 0, b->lN/2, s, 0, s_offsetN * i, s_iM, s_iN, BLOCK_SUB);
  //T_5 = A_12 - T_3     S_5 = B_22
  i = 5;
  dist_matrix_block_opp(a, 0, a->lN/2, t, 0, t_offsetN * 3, t, 0, t_offsetN * i, t_iM, t_iN, BLOCK_SUB);
  dist_matrix_block_copy(b, b->lM/2, b->lN/2, s, 0, s_offsetN * i, s_iM, s_iN);
  //T_6 = A_22           S_6 = S_3 - B_21
  i = 6;
  dist_matrix_block_copy(a, a->lM/2, a->lN/2, t, 0, t_offsetN * i, t_iM, t_iN);
  dist_matrix_block_opp(s, 0, s_offsetN * 3, b, b->lM/2, 0, s, 0, s_offsetN * i, s_iM, s_iN, BLOCK_SUB);
}

/* There are 7 of the strassen matricies to form. values: [0-6]
 */
void _caps_create_matricies_dfs(struct dbcl_t *a, struct dbcl_t *b, struct dbcl_t *c,
				struct dbcl_t *t, struct dbcl_t *s, struct dbcl_t *q,
				int strassen_step)
{
  // Initilize the datastructures for S, T, Q
  t->p = a->p; t->q = a->q;
  t->lM = a->lM/2; t->lN = a->lN/2;
  t->bm = a->bm; t->bn = a->bn;
  t->m = a->m/2; t->n = a-> n / 2;

  s->p = b->p; s->q = b->q;
  s->lM = b->lM/2; s->lN = b->lN/2;
  s->bm = b->bm; s->bn = b->bn;
  s->m = b->m/2; s->n = b-> n / 2;

  q->p = c->p; q->q = c->q;
  q->lM = c->lM/2; q->lN = c->lN/2;
  q->bm = c->bm; q->bn = c->bn;
  q->m = c->m/2; q->n = c-> n/2;

  MPI_Comm_dup(a->comm, &(t->comm));
  MPI_Comm_dup(b->comm, &(s->comm));
  MPI_Comm_dup(c->comm, &(q->comm));
  
  t->value = malloc(sizeof(double) * (t->lM * t->lN) * (t->bm * t->bn));
  if (t->value == NULL)
    exitWithError("_caps_create_matricies_dfs: malloc failed ( t->value )");

  s->value = malloc(sizeof(double) * (s->lM * s->lN) * (s->bm * s->bn));
  if (s->value == NULL)
    exitWithError("_caps_create_matricies_dfs: malloc failed ( s->value )");

  q->value = calloc((q->lM * q->lN) * (q->bm * q->bn), sizeof(double));
  if (q->value == NULL)
    exitWithError("_caps_create_matricies_dfs: calloc failed ( q->value )");
    
  // Calculate S, T according to Strassen's algorithm
  switch(strassen_step) {
  case(0):
    // T_0 = A_11   
    dist_matrix_block_copy(a, 0, 0, t, 0, 0, t->lM, t->lN);
    // S_0 = B_11
    dist_matrix_block_copy(b, 0, 0, s, 0, 0, s->lM, s->lN);
    break;
  case(1):
    // T_1 = A_12   
    dist_matrix_block_copy(a, 0, a->lN/2, t, 0, 0, t->lM, t->lN);
    // S_1 = B_21
    dist_matrix_block_copy(b, b->lM/2, 0, s, 0, 0, s->lM, s->lN);
    break;
  case(2):
    //T_2 = A_21 + A_22
    dist_matrix_block_opp(a, a->lM/2, 0, a, a->lM/2, a->lN/2, t, 0, 0, t->lM, t->lN, BLOCK_ADD);
    //S_2 = B_12 - B_11
    dist_matrix_block_opp(b, 0, b->lN/2, b, 0, 0, s, 0, 0, s->lM, s->lN, BLOCK_SUB);
    break;
  case(3):
    //T_3 = A_21 + A_22 - A_11             
    dist_matrix_block_opp(a, a->lM/2, 0, a, a->lM/2, a->lN/2, t, 0, 0, t->lM, t->lN, BLOCK_ADD);
    dist_matrix_block_opp(t, 0, 0, a, 0, 0, t, 0, 0, t->lM, t->lN, BLOCK_SUB);
    //S_3 = B_22 + B_11 - B_12
    dist_matrix_block_opp(b, b->lM/2, b->lN/2, b, 0, 0, s, 0, 0, s->lM, s->lN, BLOCK_ADD);
    dist_matrix_block_opp(s, 0, 0, b, 0, b->lN/2, s, 0, 0, s->lM, s->lN, BLOCK_SUB);
    break;
  case(4):
    //T_4 = A_11 - A_21
    dist_matrix_block_opp(a, 0, 0, a, a->lM/2, 0, t, 0, 0, t->lM, t->lN, BLOCK_SUB);
    //S_4 = B_22 - B_12
    dist_matrix_block_opp(b, b->lM/2, b->lN/2, b, 0, b->lN/2, s, 0, 0, s->lM, s->lN, BLOCK_SUB);
    break;
  case(5):
    //T_5 = A_12 + A_11 + - A_21 - A_22
    dist_matrix_block_opp(a, 0, a->lN/2, a, 0, 0, t, 0, 0, t->lM, t->lN, BLOCK_ADD);
    dist_matrix_block_opp(t, 0, 0, a, a->lM/2, 0, t, 0, 0, t->lM, t->lN, BLOCK_SUB);
    dist_matrix_block_opp(t, 0, 0, a, a->lM/2, a->lN/2, t, 0, 0, t->lM, t->lN, BLOCK_SUB);
    //S_5 = B_22
    dist_matrix_block_copy(b, b->lM/2, b->lN/2, s, 0, 0, s->lM, s->lN);
    break;
  case(6):
    //T_6 = A_22
    dist_matrix_block_copy(a, a->lM/2, a->lN/2, t, 0, 0, t->lM, t->lN);
    //S_6 = B_22 + B_11 - B_12 - B_21
    dist_matrix_block_opp(b, b->lM/2, b->lN/2, b, 0, 0, s, 0, 0, s->lM, s->lN, BLOCK_ADD);
    dist_matrix_block_opp(s, 0, 0, b, 0, b->lN/2, s, 0, 0, s->lM, s->lN, BLOCK_SUB);
    dist_matrix_block_opp(s, 0, 0, b, b->lM/2, 0, s, 0, 0, s->lM, s->lN, BLOCK_SUB);
    break;
  }
}

void _caps_add_matrix_bfs(struct dbcl_t *c, struct dbcl_t *q) {
  /* 
    Q = [ Q_0 | Q_1 | Q_2 | Q_3 | Q_4 | Q_5 | Q_6 ]
  */
  int q_iM = c->lM/2, q_iN = c->lN/2;
  int q_offsetM = 0, q_offsetN = q_iN;

  //printf("Adding to the C matrix!\n");

  // Opperations are done in specific order to minimize memory usage
  // 1. C_11 = Q_0 + Q_1
  // 2. [U_1] Q_0 = Q_0 + Q_3
  // 3. [U_2] Q_1 = [U_1] Q_0 + Q_4
  // 4. C_22 = [U_2] Q_1 + Q_2
  // 5. C_21 = [U_2] Q_1 - Q_6
  // 6. [U_3] Q_2 = [U_1] Q_0 + Q_2
  // 7. C_12 = [U_3] Q_2 + Q_5
  
  dist_matrix_block_opp(q, q_offsetM * 0, q_offsetN * 0, q, q_offsetM * 1, q_offsetN * 1, c, 0, 0, q_iM, q_iN, BLOCK_ADD);
  dist_matrix_block_opp(q, q_offsetM * 0, q_offsetN * 0, q, q_offsetM * 3, q_offsetN * 3, q, q_offsetM * 0, q_offsetN * 0, q_iM, q_iN, BLOCK_ADD);
  dist_matrix_block_opp(q, q_offsetM * 0, q_offsetN * 0, q, q_offsetM * 4, q_offsetN * 4, q, q_offsetM * 1, q_offsetN * 1, q_iM, q_iN, BLOCK_ADD);
  dist_matrix_block_opp(q, q_offsetM * 1, q_offsetN * 1, q, q_offsetM * 2, q_offsetN * 2, c, c->lM/2, c->lN/2, q_iM, q_iN, BLOCK_ADD);
  dist_matrix_block_opp(q, q_offsetM * 1, q_offsetN * 1, q, q_offsetM * 6, q_offsetN * 6, c, c->lM/2, 0, q_iM, q_iN, BLOCK_SUB);
  dist_matrix_block_opp(q, q_offsetM * 0, q_offsetN * 0, q, q_offsetM * 2, q_offsetN * 2, q, q_offsetM * 2, q_offsetN * 2, q_iM, q_iN, BLOCK_ADD);
  dist_matrix_block_opp(q, q_offsetM * 2, q_offsetN * 2, q, q_offsetM * 5, q_offsetN * 5, c, 0, c->lN/2, q_iM, q_iN, BLOCK_ADD);
}

void _caps_add_matrix_dfs(struct dbcl_t *c, struct dbcl_t *q, int strassen_step)
{
  switch(strassen_step) {
  case(0): //Q_0 in C_11, C_12, C_21, C_22
    dist_matrix_block_opp(c, 0, 0, q, 0, 0, c, 0, 0, q->lM, q->lN, BLOCK_ADD);
    dist_matrix_block_opp(c, 0, c->lN/2, q, 0, 0, c, 0, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    dist_matrix_block_opp(c, c->lM/2, 0, q, 0, 0, c, c->lM/2, 0, q->lM, q->lN, BLOCK_ADD);
    dist_matrix_block_opp(c, c->lM/2, c->lN/2, q, 0, 0, c, c->lM/2, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    break;
  case(1): //Q_1 in C_11
    dist_matrix_block_opp(c, 0, 0, q, 0, 0, c, 0, 0, q->lM, q->lN, BLOCK_ADD);
    break;
  case(2): //Q_2 in C_12, C_22
    dist_matrix_block_opp(c, 0, c->lN/2, q, 0, 0, c, 0, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    dist_matrix_block_opp(c, c->lM/2, c->lN/2, q, 0, 0, c, c->lM/2, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    break;
  case(3): //Q_3 in C_12, C_21, C_22
    dist_matrix_block_opp(c, 0, c->lN/2, q, 0, 0, c, 0, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    dist_matrix_block_opp(c, c->lM/2, 0, q, 0, 0, c, c->lM/2, 0, q->lM, q->lN, BLOCK_ADD);
    dist_matrix_block_opp(c, c->lM/2, c->lN/2, q, 0, 0, c, c->lM/2, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    break;
  case(4): //Q_4 in C_21, C_22
    dist_matrix_block_opp(c, c->lM/2, 0, q, 0, 0, c, c->lM/2, 0, q->lM, q->lN, BLOCK_ADD);
    dist_matrix_block_opp(c, c->lM/2, c->lN/2, q, 0, 0, c, c->lM/2, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    break;
  case(5): //Q_5 in C_12
    dist_matrix_block_opp(c, 0, c->lN/2, q, 0, 0, c, 0, c->lN/2, q->lM, q->lN, BLOCK_ADD);
    break;
  case(6): //-Q_6 in C_21
    dist_matrix_block_opp(c, c->lM/2, 0, q, 0, 0, c, c->lM/2, 0, q->lM, q->lN, BLOCK_ADD);
    break;
  }
}

void _caps_distribute_matricies(struct dbcl_t *t, struct dbcl_t *s, int k, MPI_Comm comm_swap, int *blockM, int *blockN) { 

  assert(t->lM == s->lM); assert(t->lN == s->lN);
  assert(t->bm == s->bm); assert(t->bn == s->bn);

  /*
    Assumes that T, S are layedout as so (this makes communication easier)
    T = [ T_0 | T_1 | T_2 | T_3 | T_4 | T_5 | T_6 ]
    S = [ S_0 | S_1 | S_2 | S_3 | S_4 | S_5 | S_6 ]
  */
  
  if (k % 2 == 1) { // Column wise
    double *temp_buffer = malloc(sizeof(double) * (t->lM * t->lN) * (t->bm * t->bn));
    if (temp_buffer == NULL)
      exitWithError("_caps_distribute_matricies: malloc failed for temp buffer( temp_buffer )");
    
    int block_size = t->lN/7 * t->lM * (t->bm * t->bn);

    int i,j,k;
    
    MPI_Alltoall(t->value, block_size, MPI_DOUBLE, temp_buffer, block_size, MPI_DOUBLE, comm_swap);
    for ( i=0; i<t->lN/7; i++ )
      for( j=0; j<t->lM; j+=*blockM )
	for ( k=0; k<7; k++ ) {
	  int send_offset = (k * t->lN/7 * t->lM + j + i * t->lM ) * (t->bm * t->bn);
	  int recv_offset = (k * *blockM + j * 7 + i * 7 * t->lM) * (t->bm * t->bn);
	  memcpy(t->value + recv_offset, temp_buffer + send_offset ,sizeof(double) * *blockM * (t->bm * t->bn));
	}

    t->lM *= 7; t->lN /= 7;

    MPI_Alltoall(s->value, block_size, MPI_DOUBLE, temp_buffer, block_size, MPI_DOUBLE, comm_swap);
    for ( i=0; i<s->lN/7; i++ )
      for( j=0; j<s->lM; j+=*blockM )
	for ( k=0; k<7; k++ ) {
	  int send_offset = (k * s->lN/7 * s->lM + j + i * s->lM ) * (s->bm * s->bn);
	  int recv_offset = (k * *blockM + j * 7 + i * 7 * s->lM) * (s->bm * s->bn);
	  memcpy(s->value + recv_offset, temp_buffer + send_offset ,sizeof(double) * *blockM * (s->bm * s->bn));
	}

    s->lM *= 7; s->lN /= 7;

    *blockM *= 7;

    free(temp_buffer);
    
  } else { // Row wise

    double *temp_buffer = malloc(sizeof(double) * (t->lM * t->lN) * (t->bm * t->bn));
    if (temp_buffer == NULL)
      exitWithError("_caps_distribute_matricies: malloc failed for temp buffer( temp_buffer )");
    
    int col_block_size = *blockN * t->lM * (t->bm * t->bn);
    int block_size = t->lN/7 * t->lM * (t->bm * t->bn);
    int stride = 7 * col_block_size;

    int i,j;
    
    MPI_Alltoall(t->value, block_size, MPI_DOUBLE, temp_buffer, block_size, MPI_DOUBLE, comm_swap);
    for(i=0; i<7; i++)
      for (j=0; j<t->lN/7; j+=*blockN )
	memcpy(t->value + j * stride + i*col_block_size, temp_buffer + i * block_size + j * col_block_size, sizeof(double) * col_block_size);

    MPI_Alltoall(s->value, block_size, MPI_DOUBLE, temp_buffer, block_size, MPI_DOUBLE, comm_swap);
    for(i=0; i<7; i++)
      for (j=0; j<s->lN/7; j+=*blockN )
	memcpy(s->value + j * stride + i*col_block_size, temp_buffer + i * block_size + j * col_block_size, sizeof(double) * col_block_size);

    *blockN *= 7;
    
    free(temp_buffer);
  }
  
}

void _caps_collect_matricies(struct dbcl_t *q, int k, MPI_Comm comm_swap, int *blockM, int *blockN) {

  /*
    Assumes that T, S are layedout as so (this makes communication easier)
    T = [ T_0 | T_1 | T_2 | T_3 | T_4 | T_5 | T_6 ]
    S = [ S_0 | S_1 | S_2 | S_3 | S_4 | S_5 | S_6 ]
  */

  int buffer_size = (q->lM * q->lN) * (q->bm * q->bn);
  double *temp_buffer = malloc(sizeof(double) * buffer_size);
  if (temp_buffer == NULL)
    exitWithError("_caps_distribute_matricies: malloc failed for temp buffer( temp_buffer )");
  
  
  if (k % 2 == 1) { // Expand Column wise

    int i,j,k;

    *blockM /= 7;
    
    q->lM /= 7; q->lN *= 7;

    int block_size = q->lN/7 * q->lM * (q->bm * q->bn);
    
    for ( i=0; i<q->lN/7; i++ )
      for( j=0; j<q->lM; j+=*blockM )
	for ( k=0; k<7; k++ ) {
	  int send_offset = (k * q->lN/7 * q->lM + j + i * q->lM ) * (q->bm * q->bn);
	  int recv_offset = (k * *blockM + j * 7 + i * 7 * q->lM) * (q->bm * q->bn);
	  assert(send_offset < buffer_size);
	  assert(recv_offset < buffer_size);
	  memcpy(temp_buffer + send_offset , q->value + recv_offset, sizeof(double) * *blockM * (q->bm * q->bn));
	}
    
    MPI_Alltoall(temp_buffer, block_size, MPI_DOUBLE, q->value, block_size, MPI_DOUBLE, comm_swap);
    
  } else { // Expand Row wise

    *blockN /= 7;
    
    int col_block_size = *blockN * q->lM * (q->bm * q->bn);
    int block_size = q->lN/7 * q->lM * (q->bm * q->bn);
    int stride = 7 * col_block_size;

    int i, j;

    for(i=0; i<7; i++)
      for (j=0; j<q->lN/7; j+=*blockN ) {
	int send_offset = j * stride + i*col_block_size;
	int recv_offset = i * block_size + j * col_block_size;
	assert(send_offset < buffer_size);
	assert(recv_offset < buffer_size);
	memcpy(temp_buffer + recv_offset, q->value + send_offset, sizeof(double) * col_block_size);
      }

    MPI_Alltoall(temp_buffer, block_size, MPI_DOUBLE, q->value, block_size, MPI_DOUBLE, comm_swap);

  }

  free(temp_buffer);
}

/* As we recurse through the bfs tree the size of joined blocks gets larger we need to keep track of this via blockM, blockN */
void _caps_bfs (struct dbcl_t *a, struct dbcl_t *b, struct dbcl_t *c, int k, int *blockM, int *blockN) {
  
  if (k == 0) {
    int bm = c->bm; int bn = c->bn;
    
    dist_matrix_to_block(a);
    dist_matrix_to_block(b);
    c->bm = a->bm; c->bn = b->bn;
    c->lM = 1; c->lN = 1;
    
    cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, a->bm, b->bn, a->bn, 1.0, a->value, a->bm, b->value, b->bm, 0.0, c->value, c->bm);
    block_to_dist_matrix(c, bm, bn);
    return;
  }

  struct dbcl_t t, s, q;
  
  _caps_create_matricies_bfs(a, b, c, &t, &s, &q, k);

  /* Create Comm group to swap T_i, S_i, and Q_i */
  int rank;
  MPI_Comm_rank(a->comm, &rank);

  MPI_Comm comm_swap;
  if (k % 2 == 1)
    MPI_Comm_split(a->comm, rank / 7, rank % 7, &comm_swap);
  else
    MPI_Comm_split(a->comm, ((rank % a->p) * a->q + rank / a->p) / 7, (rank / a->p) % 7, &comm_swap);
  
  /* Distribute matrix arrays T_i , S_i to neighbors through comm_swap*/
  _caps_distribute_matricies(&t, &s, k, comm_swap, blockM, blockN);

  _caps_bfs(&t, &s, &q, k-1, blockM, blockN);

#ifdef DEBUG_CAPS
  int debug_rank, debug_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &debug_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &debug_size);

  int i_rank;
  for(i_rank=0; i_rank<debug_size; i_rank++) {
    MPI_Barrier(MPI_COMM_WORLD);

    if (i_rank == debug_rank && i_rank == 0) {
      printf("[0] Matrix Q: k: %d\n blockM,N: %d %d\n", k, *blockM, *blockN);
      //dist_matrix_print_summary(&q);
      dist_matrix_local_print(&q);
    }
    fflush(stdout);
  }
#endif
  
  /* Distribute matrix arrays Q_i to neighbors through comm_swap*/
  _caps_collect_matricies(&q, k, comm_swap, blockM, blockN);

#ifdef DEBUG_CAPS
  MPI_Comm_rank(MPI_COMM_WORLD, &debug_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &debug_size);

  for(i_rank=0; i_rank<debug_size; i_rank++) {
    MPI_Barrier(MPI_COMM_WORLD);

    if (i_rank == debug_rank && i_rank == 0) {
      printf("[1] Matrix Q: k: %d\n", k);
      dist_matrix_print_summary(c);
      dist_matrix_local_print(&q);
      printf("[1] Matrix C Before: k: %d\n", k);
      dist_matrix_local_print(c);
    }
    fflush(stdout);
  }
#endif
    
  /* locally compute C from Q_i */
  _caps_add_matrix_bfs(c, &q);

#ifdef DEBUG_CAPS
  MPI_Comm_rank(MPI_COMM_WORLD, &debug_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &debug_size);

  for(i_rank=0; i_rank<debug_size; i_rank++) {
    MPI_Barrier(MPI_COMM_WORLD);

    if (i_rank == debug_rank && i_rank == 0) {
      printf("[2] Matrix C After: k: %d\n", k);
      dist_matrix_local_print(c);
    }
    fflush(stdout);
  }
#endif

  
  MPI_Comm_free(&comm_swap);
  dist_matrix_free(&t); dist_matrix_free(&s); dist_matrix_free(&q);
}

void _caps_dfs(struct dbcl_t *a, struct dbcl_t *b, struct dbcl_t *c, int k, double l) {

  if (l <= 0.0) {
    int blockM=1, blockN=1;
    _caps_bfs(a, b, c, k, &blockM, &blockN);
    return;
  }

  struct dbcl_t s, t, q;
  
  int i;
  for (i=0; i<7; i++) {
    /* Locally compute T_i and S_i from A and B */
    _caps_create_matricies_dfs(a, b, c, &t, &s, &q, i);

    _caps_dfs(&t, &s, &q, k, l-1.0);

    /* Compute contributions to of Q_i to C */
    _caps_add_matrix_dfs(c, &q, i);
    
    dist_matrix_free(&t); dist_matrix_free(&s); dist_matrix_free(&q);
  }
}

/* Requires:
   - a, b, c are square matricies with equal block sizes
   - s >= k + l
*/
int caps(struct dbcl_t *a, struct dbcl_t *b, struct dbcl_t *c, double local_mem_avail_mb) {
  assert(a->bm == b->bm); assert(b->bm == c->bm);
  assert(c->bn == a->bn); assert(a->bn == b->bn); assert(b->bn == c->bn);
  assert(a->m == b->m); assert(b->m == c->m);
  assert(c->m == a->n); assert(a->n == b->n); assert(b->n == c->n);

  int rank, size;
  MPI_Comm_rank(a->comm, &rank);
  MPI_Comm_size(a->comm, &size);
  
  double local_doubles_avail = (local_mem_avail_mb * 1048576.0) / (1.0 * sizeof(double));
  
  double w_0 = log2(7.0);
  double l = log2(4 * a->n / (1.0 * pow(size, 1/w_0) * sqrt(local_doubles_avail)));

  int k = ilog7(size);

  /*
  if (rank == ROOT) {
    printf("CAPS (Communication-Avoiding Parallel Strassen) Algorithm\n");
    printf("Parameters: {local_double_mem: %.0f}\n", local_doubles_avail);
    printf("DFS steps: %.2f [floor of value is taken]\n", l);
    printf("BFS steps: %d\n", k);
  }
  */
  
  _caps_dfs(a, b, c, k, l);
  
  return 0;
}

