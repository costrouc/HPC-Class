#ifndef UNITTEST_HW1_H
#define UNITTEST_HW1_H

#include <stdio.h>

void test_fail(char *file, int line, char *call, int retval);

int test_norm(unsigned int min_size, unsigned int max_size, FILE *output_file);
int test_matvecmult(unsigned int min_size, unsigned int max_size, FILE *output_file);
int test_matmatmult(unsigned int min_size, unsigned int max_size, FILE *output_file);

void print_statistics(FILE *output_file, unsigned int size, float real_time, float proc_time, long long flpins, float mflops);

#endif
