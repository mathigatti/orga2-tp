#ifndef tensorOps_h
#define tensorOps_h

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


typedef unsigned int uint;

///////////////////////////// VERSION C DOUBLE /////////////////////////////

void cost_derivative_c_double(double* vector, double* target, double* output);

void mat_plus_vec_c_double(double* matrix, double* vector, uint n, double* output);

void update_weight_c_double(double* w, double* nw, uint w_size, double c);

void hadamardProduct_c_double(double* matrix1, double* matrix2, uint n, uint m, double* output);

void matrix_prod_c_double(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output);

///////////////////////////// ASM DOUBlE /////////////////////////////

extern void cost_derivative_asm_double(double* vector, double* target, double* output);

extern void mat_plus_vec_asm_double(double* matrix, double* vector, uint n, double* output);

extern void update_weight_asm_double(double* w, double* nw, uint w_size, double c);

extern void hadamardProduct_asm_double(double* matrix1, double* matrix2, uint n, uint m, double* output);

extern void matrix_prod_asm_double(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output);

///////////////////////////// VERSION C FLOAT /////////////////////////////

void cost_derivative_c_float(float* vector, float* target, float* output);

void mat_plus_vec_c_float(float* matrix, float* vector, uint n, float* output);

void update_weight_c_float(float* w, float* nw, uint w_size, float c);

void hadamardProduct_c_float(float* matrix1, float* matrix2, uint n, uint m, float* output);

///////////////////////////// ASM FLOAT /////////////////////////////

extern void cost_derivative_asm_float(float* vector, float* target, float* output);

extern void mat_plus_vec_asm_float(float* matrix, float* vector, uint n, float* output);

extern void update_weight_asm_float(float* w, float* nw, uint w_size, float c);

extern void hadamardProduct_asm_float(float* matrix1, float* matrix2, uint n, uint m, float* output);

///////////////////////////// HELPERS DOUBLE /////////////////////////////

void vecCopy_double(double* dst, double const * src, uint size);

void printMatrix_double(double* matrix, int n, int m);

int equalVectors_double(double* v1, double* v2, uint size);

int equalMatrix_double(double* A, double* B, uint n, uint m);

void randomVector_double(uint size, double* vector, uint randMax);

void randomMatrix_double(double* matrix, uint n, uint m);

///////////////////////////// HELPERS FLOAT /////////////////////////////

void vecCopy_float(float* dst, float const * src, uint size);

void printMatrix_float(float* matrix, int n, int m);

int equalVectors_float(float* v1, float* v2, uint size);

void randomVector_float(uint size, float* vector, uint randMax);


#endif