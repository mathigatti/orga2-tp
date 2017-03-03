#include "tensorOps.h"

void cost_derivative(double* matrix, double* matrix2, uint n, uint m, double* output) {
//Return the vector of partial derivatives \partial C_x /
//partial a for the output activations.
// Normalmente m = 1
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = matrix[i * m + j] - matrix2[i * m + j];
    }
  }
}
