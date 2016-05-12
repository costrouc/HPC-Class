#include "unittests.h"
#include "test.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <papi.h>

const char *output_file_directory = "data/";

const struct test_config_t test_config[] = {
  { dgemm_unittest, "dgemm" },
  { dtrsm_unittest, "dtrsm" }
};
const unsigned int num_tests = sizeof(test_config) / sizeof(struct test_config_t);

void usage(char *executable_name)
{
  printf("%s <min_size> <max_size>\n", executable_name);
  exit(0);
}

int main(int argc, char *argv[])
{
  /* Quick check inputs are correct */
  if (argc != 3)
      usage(argv[0]);

  unsigned int min_size = atoi(argv[1]);
  unsigned int max_size = atoi(argv[2]);

  /* Check that input arguments are valid */
  if (min_size > max_size)
    usage(argv[0]);

  /* Seed random number generator */
  srand(time(NULL));

  /* Initialize PAPI Library */
  int retval;
  retval = PAPI_library_init( PAPI_VER_CURRENT );
  if ( retval != PAPI_VER_CURRENT )
    test_fail( __FILE__, __LINE__, "PAPI_library_init", retval );
  
  /* Run defined tests in test_config[]*/
  unsigned int i;
  for ( i=0; i<num_tests; i++ )
    {
      char output_file_path[128];
      strcpy(output_file_path, output_file_directory);
      strcat(output_file_path, test_config[i].test_name);
      strcat(output_file_path, ".txt");

      FILE *output_file;
      if((output_file = fopen(output_file_path, "w")) == NULL)
	{
	  fprintf(stderr, "ERROR: Failed to open output file %s for writing\n", output_file_path);
	  exit(1);
	}

      int retval;
      retval = test_config[i].test_function(min_size, max_size, output_file);
      if (retval == TEST_SUCCESS)
	printf("%s\tPASSED\n", test_config[i].test_name);
      else
	printf("%s\tFAILED\n", test_config[i].test_name);

      fclose(output_file);
    }

  /* Free all allocated PAPI recources */
  PAPI_shutdown();
  
  return 0;
}

