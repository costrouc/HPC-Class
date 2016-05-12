#include "test.h"

#include <papi.h>

#include <memory.h>
#include <stdlib.h>

void test_papi_initialize(struct test_statistics_t *test_stat)
{
  int test_counter_types[] = {
    PAPI_FP_OPS,
    PAPI_L1_DCM
  };

  test_stat->num_counters = sizeof(test_counter_types) / sizeof(int);
  
  test_stat->counters = (long long *) malloc(sizeof(long long) * test_stat->num_counters);
  
  test_stat->event_set = PAPI_NULL;

  int retval;
  if ((retval = PAPI_create_eventset((&test_stat->event_set))) != PAPI_OK )
    test_fail(__FILE__, __LINE__, "PAPI_creat_eventset", retval);
  
  if ((retval = PAPI_add_events(test_stat->event_set, test_counter_types, test_stat->num_counters)) != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_add_events", retval);

  if ((retval = PAPI_start(test_stat->event_set)) != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_start", retval);
}

void test_papi_destroy(struct test_statistics_t *test_stat)
{
  int retval;
  
  if ((retval = PAPI_stop(test_stat->event_set, test_stat->counters)) != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_stop", retval);

  if ((retval = PAPI_cleanup_eventset(test_stat->event_set)) != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);
  
  if ((retval = PAPI_destroy_eventset(&(test_stat->event_set))) != PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", retval);
}

void test_print_header(FILE *output_file)
{
  fprintf(output_file, "tid\tsize\treal_time\tfloat_ops\td1_cache_miss\tMFLOPS\n");
  fflush(output_file);
}

void test_print_statistic(int tid, unsigned int size, struct test_statistics_t test_stat, FILE *output_file)
{
  fprintf(output_file, "%d\t%d\t", tid, size);

  fprintf(output_file, "%f\t", test_stat.elapsed_time);
  
  unsigned int i;
  for ( i=0; i<test_stat.num_counters; i++ )
    fprintf( output_file, "%lld\t", test_stat.counters[i] );
  
  /* Print MFLOPS flops / usec = mflops. Counter 0 must be flops */
  fprintf( output_file, "%f\n", test_stat.counters[0] / test_stat.elapsed_time );
  fflush(output_file);
}

void test_fail(char *filename, int line, char *call, int retval){
  printf("%s\tFAILED\nLine # %d\n", filename, line);
  if ( retval == PAPI_ESYS ) {
    char buf[128];
    memset( buf, '\0', sizeof(buf) );
    sprintf(buf, "System error in %s:", call );
    perror(buf);
  }
  else if ( retval > 0 ) {
    printf("Error calculating: %s\n", call );
  }
  else {
    printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
  }
  printf("\n");
  exit(1);
}
