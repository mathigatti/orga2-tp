#include "tensorOps.h"

void update_weight(double* w, double* nw, uint w_size, uint mb_size, double eta){
/*TO OPTIMIZE*/
  for(uint i = 0; i < w_size; i++){
    w[i] -= (eta/mb_size) * nw[i];
  }
}

int max_arg(double* vector, uint n) {
  int maxIndex = 0;
  double maxValue = vector[maxIndex]; 
  for(int i = 1; i < n; i++){
    if(maxValue < vector[i]) {
      maxIndex = i;
      maxValue = vector[i];
    }
  }
  return maxIndex;
}

void transpose(double* matrix, uint n, uint m, double* output){
  /* NOTA: n y m no tienen que coincidir forzosamente con la cantidad de 
           filas y columnas real de matrix. Por ejemplo, si matrix es pxm
           con n < p, output sera la matriz que tenga por columnas las 
           primeras n filas de matrix. Esto es util a la hora de usar mini 
           batches.
  */
  for(uint i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      output[j * n + i] = matrix[i * m + j];
    }
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

void sigmoid_v(double* matrix, uint n, uint m, double* output){
/*The sigmoid function.*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++) {
      output[i * m + j] = sigmoid(matrix[i * m + j]);
    }
  }
}

void sigmoid_prime_v(double* matrix, uint n, uint m, double* output){
  double sig;
  double minusOneSig;
  for(uint i = 0; i < n; i++){ 
    for(uint j = 0; j < m; j++){
      sig = sigmoid(matrix[i * m + j]);
      minusOneSig = 1 - sig;
      output[i * m + j] = minusOneSig * sig;
    }
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

void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output){
/* matrix1 is nxm */
/* matrix2 is mxl */
/* output is nxl */
  for(uint i = 0; i < n; i++) {
    for(uint j = 0; j < l; j++){
      output[i * l + j] = 0;
      for(uint k = 0; k < m; k++){
        output[i * l + j] += matrix1[i * m + k] * matrix2[k * l + j];
      }
    }
  }
}

