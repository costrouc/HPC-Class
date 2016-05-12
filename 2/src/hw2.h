#ifndef HW1_H
#define HW1_H

void dgemm(unsigned int m, unsigned int n, unsigned int k, double *matrixa, double *matrixb, double *result);

void dtrsm(unsigned int m, unsigned int n, double *trida, double *matrixb);

#endif
