#include "tensorOps.h"

void cost_derivative(double* res_vec, double* target_mat, uint cant_imgs, double* output) {
//Return the matrix of partial derivatives \partial C_x /
//partial a for the output activations.
  for (int i = 0; i < 10; i++) {
    for (uint j = 0; j < cant_imgs; j++){
        output[i * cant_imgs + j] = res_vec[i * cant_imgs + j] - target_mat[i * cant_imgs + j];
    }
  }
}

void vector_sum(double* vector1, double* vector2, uint n, double* output){
// |vector1| == |vector2| == n
  for (int i = 0; i < n; i++) {
    output[i] = vector1[i] + vector2[i];
  }
}

void update_weight(double* w, double* nw, uint w_size, double c){
  for(uint i = 0; i < w_size; i++){
    w[i] -= c * nw[i];
  }
}

void hadamardProduct(double* matrix1, double* matrix2, uint n, uint m, double* output){
/* matrix1 and matrix2 are nxm*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = matrix1[i * m + j] * matrix2[i * m + j];
    }
  }
}


void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output){
// matrix1 is nxm
// matrix2 is mxl
// output is nxl
  for(uint i = 0; i < n; i++) {
    for(uint j = 0; j < l; j++){
      output[i * l + j] = 0;
      for(uint k = 0; k < m; k++){
        output[i * l + j] += matrix1[i * m + k] * matrix2[k * l + j];
      }
    }
  }
}
