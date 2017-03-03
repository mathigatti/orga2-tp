#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"



// Implementar en ASM

void sigmoid_v(double* matrix, uint n, uint m, double* output);

void sigmoid_prime_v(double* matrix, uint n, uint m, double* output);

void transpose(double* matrix, uint n, uint m, double* output);

int max_arg(double* vector, uint n);

void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output);

void hadamardProduct(double* matrix1, double* matrix2, uint n, uint m, double* output);

// Posiblemente se podria cambiar para que tome un vector directamente
// i.e. hacer suma de vector-vector
void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output);

// Posiblemente se podria cambiar para que tome un vector directamente
extern void cost_derivative(double* matrix, double* matrix2, uint n, uint m, double* output);

void update_weight(double* w, double* nw, uint w_size, uint mb_size, double eta);

#endif