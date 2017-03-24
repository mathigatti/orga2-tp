#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

// Implementar en ASM
// Posiblemente se podria cambiar para que tome un vector directamente
void cost_derivative(double* vector, double* target, double* output);

// Posiblemente se podria cambiar para que tome un vector directamente
// i.e. hacer suma de vector-vector
void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output);

void update_weight(double* w, double* nw, uint w_size, double c);

void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output);


#endif