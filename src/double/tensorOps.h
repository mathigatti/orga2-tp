#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

// Implementar en ASM
// Posiblemente se podria cambiar para que tome un vector directamente
void cost_derivative(double* matrix, double* matrix2, uint n, uint m, double* output);

// Posiblemente se podria cambiar para que tome un vector directamente
// i.e. hacer suma de vector-vector
void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output);



#endif