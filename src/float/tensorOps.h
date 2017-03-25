#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

// Implementar en ASM
// Posiblemente se podria cambiar para que tome un vector directamente
void cost_derivative(float* vector, float* target, float* output);

// Posiblemente se podria cambiar para que tome un vector directamente
// i.e. hacer suma de vector-vector
void mat_plus_vec(float* matrix, float* vector, uint n, uint m, float* output);

void update_weight(float* w, float* nw, uint w_size, float c);

void matrix_prod(float* matrix1, float* matrix2, uint n, uint m, uint l, float* output);


#endif