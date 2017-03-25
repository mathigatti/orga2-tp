#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

// Implementar en ASM
// Posiblemente se podria cambiar para que tome un vector directamente
void cost_derivative(float* vector, float* target, float* output);

void mat_plus_vec(float* matrix, float* vector, uint n, float* output);

void update_weight(float* w, float* nw, uint w_size, float c);

//void matrix_prod(float* matrix1, float* matrix2, uint n, uint m, uint l, float* output);

void hadamardProduct(float* matrix1, float* matrix2, uint n, uint m, float* output);

#endif