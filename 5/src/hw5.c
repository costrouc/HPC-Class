#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "papi.h"
#include "mkl_cblas.h"
#include "mkl_spblas.h"

#define MAXLINELENGTH 512
#define MAXTOKENLENGTH 128

enum sparse_format {
  SPARSE_CRS,
  SPARSE_CCS
};

enum index_format {
  INDEX_ZERO,
  INDEX_ONE
};

struct sparse_matrix_t {
  enum sparse_format s_format;
  enum index_format i_format;
  int m, n;
  int num_non_zero;
  int *primary_ind;
  int *secondary_ind;
  double *values;
};

void usage(char *executable_name)
{
  printf("Usage: %s <sparse_format> <index_format> <input_filename>\n", executable_name);
  printf("Currently supported formats:\n");
  printf("CRS - Compressed Sparse Row\n");
  printf("CCS - Compressed Sparse Column\n");
  printf("Indexing:\n");
  printf("ZERO | ONE indexing format\n");
}

void free_sparse_matrix(struct sparse_matrix_t *sparse_matrix)
{
  free(sparse_matrix->primary_ind);
  free(sparse_matrix->secondary_ind);
  free(sparse_matrix->values);
}

void print_vector(const double *vector, int m)
{
  printf("Vector:\n");
  int i;
  for ( i=0; i<m; i++ )
    printf("%lf ", vector[i] );
  printf("\n");
}

void print_sparse_matrix(const struct sparse_matrix_t sparse_matrix)
{
  int i;

  switch (sparse_matrix.s_format) {
  case (SPARSE_CRS):
    printf("Sparse Storage Format: SPARSE_CRS\n");

    printf("Row indicies:\n");
    for ( i=0; i<=sparse_matrix.m; i++ )
      printf("%d ", sparse_matrix.primary_ind[i]);
    printf("\n");

    printf("Column indicies:\n");
    for ( i=0; i<sparse_matrix.num_non_zero; i++ )
      printf("%d ", sparse_matrix.secondary_ind[i]);
    printf("\n");

    printf("Values:\n");
    for ( i=0; i<sparse_matrix.num_non_zero; i++ )
      printf("%2.1lf ", sparse_matrix.values[i]);
    printf("\n");
    break;
  case (SPARSE_CCS):
    printf("Sparse Storage Format: SPARSE_CCS\n");

    printf("Column indicies:\n");
    for ( i=0; i<=sparse_matrix.n; i++ )
      printf("%d ", sparse_matrix.primary_ind[i]);
    printf("\n");

    printf("Row indicies:\n");
    for ( i=0; i<sparse_matrix.num_non_zero; i++ )
      printf("%d ", sparse_matrix.secondary_ind[i]);
    printf("\n");

    printf("Values:\n");
    for ( i=0; i<sparse_matrix.num_non_zero; i++ )
      printf("%2.1lf ", sparse_matrix.values[i]);
    printf("\n");
    break;
  }
  
}

/* Requires:
   - Each entry in file is seperated by a space
   - Rows are in assending order
   - Columns for each row are in assending order
   - No non-zero rows
*/
int read_sparse_matrix_file(const char *input_filename, const enum sparse_format s_format, const enum index_format i_format, struct sparse_matrix_t *sparse_matrix)
{
  int m, n;
  int num_non_zero;
  int retval;

  /* Open the inputfile */
  FILE *input_file;
  if ((input_file = fopen(input_filename, "r")) == NULL)
    {
      fprintf(stderr, "Error: unable to crs file: %s\n", input_filename);
      return 1;
    }

  /* Read the header: <num_rows> <num_columns> <num_non_zero> */
  retval = fscanf(input_file, "%d %d %d", &m, &n, &num_non_zero);
  if (retval != 3)
    {
      fprintf(stderr, "Error: improper header, correct usage: <num_rows> <num_columns> <num_non_zero>\n");
      return 1;
    }

  sparse_matrix->s_format = s_format;
  sparse_matrix->m = m;
  sparse_matrix->n = n;
  sparse_matrix->num_non_zero = num_non_zero;

  if (sparse_matrix->s_format == SPARSE_CRS)
      sparse_matrix->primary_ind = (int *) malloc(sizeof(int) * (m + 1));
  else if (sparse_matrix->s_format == SPARSE_CCS)
      sparse_matrix->primary_ind = (int *) malloc(sizeof(int) * (n + 1));
  
  sparse_matrix->secondary_ind = (int *) malloc(sizeof(int) * num_non_zero);
  sparse_matrix->values = (double *) malloc(sizeof(double) * num_non_zero);

  int row_prev = 0, col_prev = 0;
  int entry_index = 0;
  int row_index, col_index;
  double value;
  
  while (retval = fscanf(input_file, "%d %d %lf", &row_index, &col_index, &value) != EOF)
    {
      if (i_format == INDEX_ZERO)
	row_index = row_index;
      else
	row_index = row_index - 1;
      
      switch (sparse_matrix->s_format) {
      case (SPARSE_CRS):
	/* Check for potential errors in crs input file line */
	if (row_index < row_prev)
	  {
	    fprintf(stderr, "Error: rows must be in assending order\n");
	    fclose(input_file);
	    return 1;
	  }
	else if (row_index == row_prev && col_index < col_prev)
	  {
	    fprintf(stderr, "Error: columns in each row must be in assending order\n");
	    fclose(input_file);
	    return 1;
	  }
	else if (entry_index == num_non_zero)
	  {
	    fprintf(stderr, "Error: non-zeros specified exceed stated number of non-zeros in header\n");
	    fclose(input_file);
	    return 1;
	  }
      
	/* Insert value into table */
	if (row_index != row_prev) 
	  {
	    int i;
	    for ( i=(row_prev+1); i<=row_index; i++ )
	      sparse_matrix->primary_ind[i] = entry_index;
	  }

	if (i_format == INDEX_ZERO)
	  sparse_matrix->secondary_ind[entry_index] = col_index;
	else
	  sparse_matrix->secondary_ind[entry_index] = col_index - 1;
	
	sparse_matrix->values[entry_index] = value;
	break;
	  
      case (SPARSE_CCS):
	/* Check for potential errors in crs input file line */
	if (col_index < col_prev)
	  {
	    fprintf(stderr, "Error: columns must be in assending order\n");
	    fclose(input_file);
	    return 1;
	  }
	else if (col_index == col_prev && row_index < row_prev)
	  {
	    fprintf(stderr, "Error: rows in each column must be in assending order\n");
	    fclose(input_file);
	    return 1;
	  }
	else if (entry_index == num_non_zero)
	  {
	    fprintf(stderr, "Error: non-zeros specified exceed stated number of non-zeros in header\n");
	    fclose(input_file);
	    return 1;
	  }

	/* Insert value into table */
	if (col_index != col_prev) 
	  {
	    int i;
	    for ( i=(col_prev+1); i<=col_index; i++ )
	      sparse_matrix->primary_ind[i] = entry_index;
	  }

	if (i_format == INDEX_ZERO)
	  sparse_matrix->secondary_ind[entry_index] = row_index;
	else
	  sparse_matrix->secondary_ind[entry_index] = row_index - 1;
	sparse_matrix->values[entry_index] = value;
      }
      
      entry_index++;

      row_prev = row_index;
      col_prev = col_index;
    }

  int i;
  switch (sparse_matrix->s_format) {
  case (SPARSE_CCS):
    for ( i=(col_prev+1); i<=n; i++ )
      sparse_matrix->primary_ind[i] = entry_index;
    break;
  case (SPARSE_CRS):
    for (i=(row_prev+1); i<=m; i++ )
      sparse_matrix->primary_ind[i] = entry_index;
    break;
  }
    
  if (entry_index != num_non_zero)
    {
      fprintf(stderr, "Error: non-zeros specified do not equal stated number of non-zeros in header\n");
      fclose(input_file);
      return 1;
    }

  fclose(input_file);

  return 0;
}

int sparse_dgemv(const struct sparse_matrix_t sparse_matrix, const double *vector, double *result)
{
  int i, j;

  switch (sparse_matrix.s_format) {
  case (SPARSE_CRS):
    for ( i=0; i<sparse_matrix.m; i++ )
      {
	int start_index = sparse_matrix.primary_ind[i];
	int end_index = sparse_matrix.primary_ind[i+1];
	
	result[i] = 0.0;
	
	for ( j=start_index; j<end_index; j++ )
	  {
	    result[i] += sparse_matrix.values[j] * vector[sparse_matrix.secondary_ind[j]];
	  }
      }
    break;
  case (SPARSE_CCS):
    for ( i=0; i<sparse_matrix.m; i++ )
      result[i] = 0.0;
    for ( j=0; j<sparse_matrix.n; j++ )
      {
	int start_index = sparse_matrix.primary_ind[j];
	int end_index = sparse_matrix.primary_ind[j+1];
	
	for ( i=start_index; i<end_index; i++ )
	  {
	    result[sparse_matrix.secondary_ind[i]] += sparse_matrix.values[i] * vector[sparse_matrix.secondary_ind[i]];
	  }
      }
    break;
  }

  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 4)
    {
      usage(argv[0]);
      return 1;
    }

  /* Determine the storage format */
  enum sparse_format s_format;
  if (strcmp("CRS", argv[1]) == 0)
    s_format = SPARSE_CRS;
  else if (strcmp("CCS", argv[1]) == 0)
    s_format = SPARSE_CCS;
  else
    {
      fprintf(stderr, "Error: Storage format %s unsupported\n", argv[1]);
      return 1;
    }

  /* Determine the storage format */
  enum index_format i_format;
  if (strcmp("0", argv[2]) == 0)
    i_format = INDEX_ZERO;
  else if (strcmp("1", argv[2]) == 0)
    i_format = INDEX_ONE;
  else
    {
      fprintf(stderr, "Error: index format %s unsupported\n", argv[2]);
      return 1;
    }

  
  struct sparse_matrix_t sparse_matrix;
  int retval;
  
  if((retval = read_sparse_matrix_file(argv[3], s_format, i_format, &sparse_matrix)) == 1)
    {
      fprintf(stderr, "There was an Error reading %s\n", argv[3]);
      return 1;
    }

  //print_sparse_matrix(sparse_matrix);

  double *vector = (double *) malloc( sizeof(double) * sparse_matrix.n );
  int i;
  for ( i=0; i<sparse_matrix.n; i++ )
    vector[i] = 1.0;//rand() / (double) RAND_MAX;
  
  double *result = (double *) malloc( sizeof(double) * sparse_matrix.m );
  double *result_mkl = (double *) malloc( sizeof(double) * sparse_matrix.m );
  
  float real_start_time, proc_start_time, mflops_start;
  float real_end_time, proc_end_time, mflops_end;
  long long flpins_start, flpins_end;

  if((retval=PAPI_flops( &real_start_time, &proc_start_time, &flpins_start, &mflops_start))<PAPI_OK)
      return 1;
  
  sparse_dgemv(sparse_matrix, vector, result);

  /* Collect the data into the variables passed in */
  if((retval=PAPI_flops( &real_end_time, &proc_end_time, &flpins_end, &mflops_end))<PAPI_OK)
      return 1;

  /* Print header information */
  printf("M\tN\tNum Non Zeros\tReal_Time [s]\tProc_Time [s]\tFLOPS\tMFLOPS\n");
  
  /* print statistics to stdout */
  printf("%d\t%d\t%d\t", sparse_matrix.m, sparse_matrix.n, sparse_matrix.num_non_zero);
  printf("%e\t%e\t", real_end_time - real_start_time, proc_end_time - proc_start_time);
  printf("%lld\t%f\n", flpins_end - flpins_start, mflops_end);

  /* We can check our answer is we used the CRS format using mkl*/
  if (s_format == SPARSE_CRS)
    {
      char trans = 'N';
      mkl_cspblas_dcsrgemv(&trans, &(sparse_matrix.m), sparse_matrix.values, sparse_matrix.primary_ind, sparse_matrix.secondary_ind, vector, result_mkl);

      double norm;
      /* Calculate difference between two matricies */
      for ( i=0; i<sparse_matrix.m; i++ )
	result[i] = result[i] - result_mkl[i];
      
      norm = cblas_dnrm2(sparse_matrix.m, result, 1);
      
      printf("Norm difference between matricies %lf\n", norm);
    }

  free_sparse_matrix(&sparse_matrix);
  
  return 0;
}
