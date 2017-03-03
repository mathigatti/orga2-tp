#ifndef tensorOps_h
#define tensorOps_h
#include "helpers.h"

// Implementar en ASM
// Posiblemente se podria cambiar para que tome un vector directamente
extern void cost_derivative(double* matrix, double* matrix2, uint n, uint m, double* output);

#endif