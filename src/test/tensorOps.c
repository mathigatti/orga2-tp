#include "tensorOps.h"

void cost_derivative_c(double* res_vec, double* target_vec, double* output) {
//Return the vector of partial derivatives \partial C_x /
//partial a for the output activations.
// Normalmente m = 1
  for(uint i = 0; i < 10; i++){
      output[i] = res_vec[i] - target_vec[i];
  }
}

void mat_plus_vec_c(double* matrix, double* vector, uint n, uint m, double* output){
// |vector| == n

  for(int i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = vector[i] + matrix[i * m + j];
    }
  }
}

void update_weight_c(double* w, double* nw, uint w_size, double c){
  for(uint i = 0; i < w_size; i++){
    w[i] -= c * nw[i];
  }
}


void printMatrix(double* matrix, int n, int m) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      printf("%.3f ", matrix[i * m + j]);
    }
    printf("\n");
  }
}


int equalVectors(double* v1, double* v2, uint size){
  for (uint i = 0; i < size; i++){
    if (v1[i] != v2[i]){
      return 0;
    }
  }
  return 1;
};

void randomVector(uint size, double* vector, uint randMax){

  for (uint i = 0; i < size; i++){
      vector[i] = (double) rand() / RAND_MAX;
  }

};

void vecCopy(double* dst, double const * src, uint size){
  for (uint i = 0; i < size; i++){
      dst[i] = src[i];
  }

}

