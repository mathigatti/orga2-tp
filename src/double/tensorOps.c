#include "tensorOps.h"

void cost_derivative(double* res_vec, double* target_vec, double* output) {
//Return the vector of partial derivatives \partial C_x /
//partial a for the output activations.
// Normalmente m = 1
  for(uint i = 0; i < 10; i++){
      output[i] = res_vec[i] - target_vec[i];
  }
}

void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output){
// |vector| == n

  for(int i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = vector[i] + matrix[i * m + j];
    }
  }
}

void update_weight(double* w, double* nw, uint w_size, double c){
  for(uint i = 0; i < w_size; i++){
    w[i] -= c * nw[i];
  }
}


