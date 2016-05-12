#include "unittest_hw1.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MINSIZE 1
#define MAXSIZE 10000

void usage(char *executable_name)
{
  printf("%s <min_size> <max_size>\n", executable_name);
  exit(0);
}

int main(int argc, char *argv[])
{
  // Quick check inputs are correct
  if (argc != 3)
      usage(argv[0]);

  unsigned int min_size = atoi(argv[1]);
  unsigned int max_size = atoi(argv[2]);

  int retval;
  FILE *output_file;

  /* Seed random number generator */
  srand(time(NULL));
  
  /* Vector Norm */
  if((output_file = fopen("data/norm.txt", "w")) == NULL)
    {
      printf("ERROR: Failed to open output file for writing\n");
      exit(1);
    }
  retval = test_norm(min_size, max_size, output_file);
  if (retval == 0)
    printf("norm\tPASSED\n");
  else
    printf("norm\tFAILED\n");
  
  fclose(output_file);

  
  /* Vector Matrix Product */
  if((output_file = fopen("data/matvec.txt", "w")) == NULL)
    {
      printf("ERROR: Failed to open output file for writing\n");
      exit(1);
    }
  retval = test_matvecmult(min_size, max_size, output_file);
  
  if (retval == 0)
    printf("matvecmult\tPASSED\n");
  else
    printf("matvecmult\tFAILED\n");
  
  fclose(output_file);

  
  /* Matrix Matrix Product */
  if((output_file = fopen("data/matmat.txt", "w")) == NULL)
    {
      printf("ERROR: Failed to open output file for writing\n");
      exit(1);
    }
  retval = test_matmatmult(min_size, max_size, output_file);
  
  if (retval == 0)
    printf("matmatmult\tPASSED\n");
  else
    printf("matmatmult\tFAILED\n");
  
  fclose(output_file);
  
  return 0;
}

