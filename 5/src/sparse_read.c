#define BLOCK_SIZE 3

enum sparse_format {
  SPARSE_CRS,
  SPARSE_CCS,
  SPARSE_BRS
};

enum index_format {
  INDEX_ZERO = 0,
  INDEX_ONE = 1
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

int read_aij_file(const char *input_filename, const enum sparse_format sparse_output_format, const enum index_format index_input_format, const enum index_format index_output_format, struct sparse_matrix_t *sparse_matrix)
{
  int retval;
  
  /* Open the inputfile */
  FILE *input_file;
  if ((input_file = fopen(input_filename, "r")) == NULL) {
      fprintf(stderr, "Error: unable read aij file: %s\n", input_filename);
      return 1;
  }

  /* Read the header: <num_rows> <num_columns> <num_non_zero> */
  retval = fscanf(input_file, "%d %d %d", &m, &n, &num_non_zero);
  if (retval != 3) {
      fprintf(stderr, "Error: improper header, correct usage: <num_rows> <num_columns> <num_non_zero>\n");
      return 1;
  }


  int *coor_rows = (int *) malloc( sizeof(int) * num_non_zeros );
  int *coor_columns = (int *) malloc( sizeof(int) * num_non_zeros );
  double *coor_values = (double *) malloc( sizeof(double) * num_non_zeros ); 
    
  int i = 0;
  int row_index, col_index;
  double value;
  
  while ( retval = fscanf(input_file, "%d %d %lf", &row_index, &col_index, &value) != EOF ) {
    if (retval != 3) {
      fprintf(stderr, "Error: failed to read three values\n");
      return 1;
    }

    if ( i == num_non_zeros ) {
      fprintf("Error: stated %d non-zero entries but more are supplied\n");
      return 1;
    }

    coor_rows[i].i = row_index;
    coor_columns[i].j = col_index;
    coor_values[i].value = value;

    i++;
  }

  if ( i == num_non_zeros ) {
      fprintf("Error: stated %d non-zero entries but less were supplied\n");
      return 1;
  }
  
  /* Convert Coordinate to CRS format */
  int crs_job[] = { 1, index_output_format, index_input_format, 0, num_non_zeros, 0 };
  
  crs_rows = malloc( sizeof(int) * ( sparse_matrix.m + 1 ));
  crs_columns = malloc( sizeof(int) * num_non_zeros );
  crs_values = malloc( sizeof(double) * num_non_zeros );

  int info;
  
  mkl_dcsrcoo (crs_job, m, csr_values, csr_columns, csr_rows, *num_non_zeros, coor_values, coor_rows, coor_columns, *info);

  if ( sparse_output_format == SPARSE_BRS ) {
    int brs_job =  { 0, index_output_format, index_output_format, 0, 0, 1 };

    /* Be safe and allocate block_size ^ 2 * num_non_zeros */
    brs_values = malloc(sizeof(double) * BLOCK_SIZE * BLOCK_SIZE * num_non_zeros );
    brs_rows = malloc(sizeof(
    
    mkl_dcsrbsr (brs_job, m, &(int){ BLOCK_SIZE }, &(int){ BLOCK_SIZE }, csr_values, csr_columns, csr_rows, double *Absr, MKL_INT * AJB, MKL_INT * AIB, MKL_INT * info);
  }
  
  sparse_matrix->s_format = sparse_output_format;
  sparse_matrix->i_format = index_output_format;
  sparse_matrix->m = m;
  sparse_matrix->n = n;
  sparse_matrix->num_non_zero = num_non_zero;

  
}
