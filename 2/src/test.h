#ifndef TEST_H
#define TEST_H

#include <stdio.h>
#include <papi.h>

#define MACH_ERROR 1E-10

#define TEST_SUCCESS 0
#define TEST_FAIL 1

struct test_config_t {
  int (*test_function)(unsigned int min_size, unsigned int max_size, FILE *output_file);
  char *test_name;
};


struct test_statistics_t {
  int event_set;
  double elapsed_time;
  unsigned int num_counters;
  long long *counters;
};

void test_papi_initialize(struct test_statistics_t *test_stat);
void test_papi_destroy(struct test_statistics_t *test_stat);
void test_fail(char *filename, int line, char *call, int retval);
void test_print_header(FILE *output_file);
void test_print_statistic(int tid, unsigned int size, struct test_statistics_t test_stat, FILE *output_file);

#endif


  
