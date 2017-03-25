#include "tensorOps.h"

///////////// VERSION C DOUBLE /////////////

void cost_derivative_c_double(double* res_vec, double* target_vec, double* output) {
//Return the vector of partial derivatives \partial C_x /
//partial a for the output activations.
// Normalmente m = 1
  for(uint i = 0; i < 10; i++){
      output[i] = res_vec[i] - target_vec[i];
  }
}

void mat_plus_vec_c_double(double* matrix, double* vector, uint n, double* output){
// |vector| == n
  for(uint i = 0; i < n; i++){
    output[i] = vector[i] + matrix[i];
  }
}

void update_weight_c_double(double* w, double* nw, uint w_size, double c){
  for(uint i = 0; i < w_size; i++){
    w[i] -= c * nw[i];
  }
}

void hadamardProduct_c_double(double* matrix1, double* matrix2, uint n, uint m, double* output){
/* matrix1 and matrix2 are nxm*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = matrix1[i * m + j] * matrix2[i * m + j];
    }
  }
}

///////////// VERSION C FLOAT /////////////

void cost_derivative_c_float(float* res_vec, float* target_vec, float* output) {
//Return the vector of partial derivatives \partial C_x /
//partial a for the output activations.
// Normalmente m = 1
  for(uint i = 0; i < 10; i++){
      output[i] = res_vec[i] - target_vec[i];
  }
}

void mat_plus_vec_c_float(float* matrix, float* vector, uint n, float* output){
// |vector| == n
  for(uint i = 0; i < n; i++){
    output[i] = vector[i] + matrix[i];
  }
}

void update_weight_c_float(float* w, float* nw, uint w_size, float c){
  for(uint i = 0; i < w_size; i++){
    w[i] -= c * nw[i];
  }
}

void hadamardProduct_c_float(float* matrix1, float* matrix2, uint n, uint m, float* output){
/* matrix1 and matrix2 are nxm*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = matrix1[i * m + j] * matrix2[i * m + j];
    }
  }
}

///////////// HELPERS DOUBLE /////////////

void printMatrix_double(double* matrix, int n, int m) {
  for(uint i = 0; i < n; i++) {
    for(uint j = 0; j < m; j++) {
      printf("%.3f ", matrix[i * m + j]);
    }
    printf("\n");
  }
}


int equalVectors_double(double* v1, double* v2, uint size){
  for (uint i = 0; i < size; i++){
    if (v1[i] != v2[i]){
      return 0;
    }
  }
  return 1;
}

void randomVector_double(uint size, double* vector, uint randMax){

  for (uint i = 0; i < size; i++){
      vector[i] = (double) rand() / RAND_MAX;
  }
}

void vecCopy_double(double* dst, double const * src, uint size){
  for (uint i = 0; i < size; i++){
      dst[i] = src[i];
  }
}

///////////// HELPERS FLOAT /////////////

void printMatrix_float(float* matrix, int n, int m) {
  for(uint i = 0; i < n; i++) {
    for(uint j = 0; j < m; j++) {
      printf("%.3f ", matrix[i * m + j]);
    }
    printf("\n");
  }
}


int equalVectors_float(float* v1, float* v2, uint size){
  for (uint i = 0; i < size; i++){
    if (v1[i] != v2[i]){
      return 0;
    }
  }
  return 1;
}

void randomVector_float(uint size, float* vector, uint randMax){
  for (uint i = 0; i < size; i++){
      vector[i] = (float) rand() / RAND_MAX;
  }
}

void vecCopy_float(float* dst, float const * src, uint size){
  for (uint i = 0; i < size; i++){
      dst[i] = src[i];
  }
}