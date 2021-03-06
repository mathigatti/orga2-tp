#include <assert.h>
#include "tensorOps.h"
#include <unistd.h>

#define SIZE 10
#define NUM_DIGITS 10
#define randMax 10
#define c 0.1

int main(){
  printf("TESTEANDO DOUBLES\n");

  double v[SIZE] = {[0 ... SIZE-1] = 0};
  double y[SIZE] = {[0 ... SIZE-1] = 0};
  double w_asm[SIZE] = {[0 ... SIZE-1] = 0};
  double w_c[SIZE] = {[0 ... SIZE-1] = 0};
  double res_c[SIZE] = {[0 ... SIZE-1] = 0};
  double res_asm[SIZE] = {[0 ... SIZE-1] = 0};

  uint n = 10;
  uint m = 20;
  uint l = 30;
  double A[n * m];
  double B[m * l];
  double C_c[n * l];
  double C_asm[n * l];

  srand(time(NULL));

  printf("%s\n", "\tTesteo matrix_prod...");

  for(uint i = 0; i < 100; i++){

    randomMatrix_double(A, n, m);

    randomMatrix_double(B, m, l);


    matrix_prod_c_double(A, B, n, m, l, C_c);
    matrix_prod_asm_double(A, B, n, m, l, C_asm);
    
    assert(equalMatrix_double(C_c, C_asm, n, l));
  }

  printf("%s\n", "\t\tTests pasados exitosamente por matrix_prod");

  printf("%s\n", "\tTesteo cost_derivative...");

  for(uint i = 0; i < 100; i++){

    randomVector_double(SIZE, v, randMax);
    randomVector_double(SIZE, y, randMax);

    cost_derivative_c_double(v, y, 1, res_c);
    cost_derivative_asm_double(v, y, res_asm);

    assert(equalVectors_double(res_c,res_asm,SIZE));
  }

  printf("%s\n", "\t\tTests pasados exitosamente por cost_derivative");


  printf("%s\n", "\tTesteo mat_plus_vec...");

  for(uint i = 0; i < 100; i++){

    randomVector_double(SIZE, v, randMax);
    randomVector_double(SIZE, y, randMax);

    mat_plus_vec_c_double(v, y, SIZE, 1, res_c);
    mat_plus_vec_asm_double(v, y, SIZE, res_asm);

    assert(equalVectors_double(res_c,res_asm,SIZE));

  }

  printf("%s\n", "\t\tTests pasados exitosamente por mat_plus_vec");

  printf("%s\n", "\tTesteo update_weight...");

  for(uint i = 0; i < 100; i++){

    randomVector_double(SIZE, w_asm, randMax);
    randomVector_double(SIZE, y, randMax);
    vecCopy_double(w_c, w_asm, SIZE);
    
    update_weight_c_double(w_c, y, SIZE, c);
    update_weight_asm_double(w_asm, y, SIZE, c);

    assert(equalVectors_double(w_c,w_asm,SIZE));

  }

  printf("%s\n", "\t\tTests pasados exitosamente por update_weight");

  printf("%s\n", "\tTesteo hadamardProduct...");

  for(uint i = 0; i < 100; i++){

    randomVector_double(SIZE, v, randMax);
    randomVector_double(SIZE, y, randMax);

    hadamardProduct_c_double(v, y, SIZE, 1, res_c);
    hadamardProduct_asm_double(v, y, SIZE, 1, res_asm);
    
    assert(equalVectors_double(res_c,res_asm,SIZE));

  }

  printf("%s\n", "\t\tTests pasados exitosamente por hadamardProduct");

  printf("TODOS LOS TESTS DE DOUBLES PASARON EXITOSAMENTE\n\n");

  printf("TESTEANDO FLOATS\n");

  float v_float[SIZE] = {[0 ... SIZE-1] = 0};
  float y_float[SIZE] = {[0 ... SIZE-1] = 0};
  float w_asm_float[SIZE] = {[0 ... SIZE-1] = 0};
  float w_c_float[SIZE] = {[0 ... SIZE-1] = 0};
  float res_c_float[SIZE] = {[0 ... SIZE-1] = 0};
  float res_asm_float[SIZE] = {[0 ... SIZE-1] = 0};

//  uint nf = 30;
//  uint mf = 784;
//  uint lf = 1;

  uint nf = 10;
  uint mf = 30;
  uint lf = 1;

//  uint nf = 10;
//  uint mf = 3;
//  uint lf = 10;

  float Af[nf * mf];
  float Bf[mf * lf];
  float Cf_c[nf * lf];
  float Cf_asm[nf * lf];

  srand(time(NULL));

  printf("%s\n", "\tTesteo matrix_prod...");

  for(uint i = 0; i < 1; i++){

    randomMatrix_float(Af, nf, mf);

    randomMatrix_float(Bf, mf, lf);

    matrix_prod_c_float(Af, Bf, nf, mf, lf, Cf_c);
    matrix_prod_asm_float(Af, Bf, nf, mf, lf, Cf_asm);

    assert(equalMatrix_float(Cf_c, Cf_asm, nf, lf));
  }

  printf("%s\n", "\tTesteo cost_derivative...");

  for(uint i = 0; i < 100; i++){

    randomVector_float(NUM_DIGITS, v_float, randMax);
    randomVector_float(NUM_DIGITS, y_float, randMax);

    cost_derivative_c_float(v_float, y_float, res_c_float);
    cost_derivative_asm_float(v_float, y_float, res_asm_float);

    assert(equalVectors_float(res_c_float,res_asm_float,NUM_DIGITS));
  }

  printf("%s\n", "\t\tTests pasados exitosamente por cost_derivative");

  printf("%s\n", "\tTesteo mat_plus_vec...");

  for(uint i = 0; i < 100; i++){

    randomVector_float(SIZE, v_float, randMax);
    randomVector_float(SIZE, y_float, randMax);

    mat_plus_vec_c_float(v_float, y_float, SIZE, res_c_float);
    mat_plus_vec_asm_float(v_float, y_float, SIZE, res_asm_float);

    assert(equalVectors_float(res_c_float,res_asm_float,SIZE));

  }

  printf("%s\n", "\t\tTests pasados exitosamente por mat_plus_vec");

  printf("%s\n", "\tTesteo hadamardProduct...");

  for(uint i = 0; i < 100; i++){

    randomVector_float(SIZE, v_float, randMax);
    randomVector_float(SIZE, y_float, randMax);

    hadamardProduct_c_float(v_float, y_float, SIZE, 1, res_c_float);
    hadamardProduct_asm_float(v_float, y_float, SIZE, 1, res_asm_float);
    
    assert(equalVectors_float(res_c_float,res_asm_float,SIZE));

  }

  printf("%s\n", "\t\tTests pasados exitosamente por hadamardProduct");

  printf("%s\n", "\tTesteo update_weight...");

  for(uint i = 0; i < 100; i++){

    randomVector_float(SIZE, w_asm_float, randMax);
    randomVector_float(SIZE, y_float, randMax);
    vecCopy_float(w_c_float, w_asm_float, SIZE);
    
    update_weight_c_float(w_c_float, y_float, SIZE, c);
    update_weight_asm_float(w_asm_float, y_float, SIZE, c);

    assert(equalVectors_float(w_c_float, w_asm_float, SIZE));

  }

  printf("%s\n", "\t\tTests pasados exitosamente por update_weight");

  printf("TODOS LOS TESTS DE FLOAT PASARON EXITOSAMENTE\n\n");


  return 0;
}
