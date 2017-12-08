#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

// Metodos implementados tanto en C como en ASSEMBLER
void cost_derivative(double* vector, double* target, uint cant_imgs, double* output);

void vector_sum(double* vector1, double* vector2, uint n, double* output);

void update_weight(double* w, double* nw, uint w_size, double c);

void hadamard_product(double* matrix1, double* matrix2, uint n, uint m, double* output);

void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output);

#endif