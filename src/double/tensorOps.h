#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

void cost_derivative(double* vector, double* target, uint cant_imgs, double* output);

void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output);

void update_weight(double* w, double* nw, uint w_size, double c);

void hadamardProduct(double* matrix1, double* matrix2, uint n, uint m, double* output);

void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output);

#endif