#ifndef tensorOps_h
#define tensorOps_h

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


typedef unsigned int uint;

void vecCopy(double* dst, double const * src, uint size);

void printMatrix(double* matrix, int n, int m);

int equalVectors(double* v1, double* v2, uint size);

void randomVector(uint size, double* vector, uint randMax);

double sigmoid(double z);

double sigmoid_prime(double z);


////////////////// VERSION C /////////////////////

void cost_derivative_c(double* vector, double* target, double* output);

void mat_plus_vec_c(double* matrix, double* vector, uint n, uint m, double* output);

void update_weight_c(double* w, double* nw, uint w_size, double c);

void sigmoid_v_c(double* matrix, uint n, uint m, double* output);

void sigmoid_prime_v_c(double* matrix, uint n, uint m, double* output);

void transpose_c(double* matrix, uint n, uint m, double* output);

void hadamardProduct_c(double* matrix1, double* matrix2, uint n, uint m, double* output);

///////////////// VERSION ASM /////////////////////////////

extern void cost_derivative_asm(double* vector, double* target, double* output);

extern void mat_plus_vec_asm(double* matrix, double* vector, uint n, uint m, double* output);

extern void update_weight_asm(double* w, double* nw, uint w_size, double c);

extern void hadamardProduct_asm(double* matrix1, double* matrix2, uint n, uint m, double* output);

#endif