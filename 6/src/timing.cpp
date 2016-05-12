#include <sys/time.h>
typedef struct timestruct
{
  unsigned int sec;
  unsigned int usec;
} TimeStruct;

/* ////////////////////////////////////////////////////////////////////////////
   -- Get current time
*/ 
extern "C"
TimeStruct get_current_time(void)
{
  static struct timeval  time_val;
  static struct timezone time_zone;

  TimeStruct time;

  cudaThreadSynchronize();
  gettimeofday(&time_val, &time_zone);

  time.sec  = time_val.tv_sec;
  time.usec = time_val.tv_usec;
  return (time);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- End elapsed time
*/ 
extern "C"
double GetTimerValue(TimeStruct time_1, TimeStruct time_2)
{
  int sec, usec;

  sec  = time_2.sec  - time_1.sec;
  usec = time_2.usec - time_1.usec;

  return (1000.*(double)(sec) + (double)(usec) * 0.001);
}


