#ifndef UNITTESTS_H
#define UNITTESTS_H

#include <stdio.h>

int dgemm_unittest(unsigned int min_size, unsigned int max_size, FILE *output_file);
int dtrsm_unittest(unsigned int min_size, unsigned int max_size, FILE *output_file);

#endif
