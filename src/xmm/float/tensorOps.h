#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

// Metodos implementados tanto en C como en ASSEMBLER
void cost_derivative(float* vector, float* target, float* output);

void mat_plus_vec(float* matrix, float* vector, uint n, float* output);

void update_weight(float* w, float* nw, uint w_size, float c);

void hadamardProduct(float* matrix1, float* matrix2, uint n, uint m, float* output);

void matrix_prod(float* matrix1, float* matrix2, uint n, uint m, uint l, float* output);

#endif