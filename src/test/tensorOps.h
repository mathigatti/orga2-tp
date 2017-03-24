#ifndef tensorOps_h
#define tensorOps_h

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>


typedef unsigned int uint;

void vecCopy(double* dst, double const * src, uint size);

void printMatrix(double* matrix, int n, int m);

int equalVectors(double* v1, double* v2, uint size);

void randomVector(uint size, double* vector, uint randMax);


////////////////// VERSION C /////////////////////

void cost_derivative_c(double* vector, double* target, double* output);

void mat_plus_vec_c(double* matrix, double* vector, uint n, uint m, double* output);

void update_weight_c(double* w, double* nw, uint w_size, double c);

///////////////// VERSION ASM /////////////////////////////

extern void cost_derivative_asm(double* vector, double* target, double* output);

extern void mat_plus_vec_asm(double* matrix, double* vector, uint n, uint m, double* output);

extern void update_weight_asm(double* w, double* nw, uint w_size, double c);

#endif