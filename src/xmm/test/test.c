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

  uint n = 200;
  uint m = 100;
  uint l = 150;
  double A[n * m];
  double B[m * l];
  double C_c[n * l];
  double C_asm[n * l];

  uint n_2 = 3;
  uint m_2 = 2;
  uint l_2 = 5;
  double A_2[n * m];
  double B_2[m * l];
  double C_c_2[n * l];
  double C_asm_2[n * l];

  srand(time(NULL));

  printf("%s\n", "\tTesteo matrix_prod...");

  for(uint i = 0; i < 1; i++){

    noRandomMatrix_double(A_2, n_2, m_2, 5);

    noRandomMatrix_double(B_2, m_2, l_2, 2);


    matrix_prod_c_double(A_2, B_2, n_2, m_2, l_2, C_c_2);
    matrix_prod_asm_double(A_2, B_2, n_2, m_2, l_2, C_asm_2);

    printMatrix_double(A_2, n_2, m_2);

    printf("\n");

    printMatrix_double(B_2, m_2, l_2);

    printf("\n");

    printMatrix_double(C_asm_2, n_2, l_2);

    assert(equalMatrix_double(C_c_2, C_asm_2, n_2, l_2));
    //Dato copado: usando un criterio de != el assert falla,
    //pero con un umbral de 10^(-10) [y posiblemente bastante menos tambien]
    //pasa los test exitosamente. Esto se debe casi seguro a diferencias 
    //por underflow (debido a que el orden de las sumas varia en c y asm)

  }

  printf("%s\n", "\t\tTests pasados exitosamente por matrix_prod");

  printf("%s\n", "\tTesteo cost_derivative...");

  for(uint i = 0; i < 100; i++){

    randomVector_double(SIZE, v, randMax);
    randomVector_double(SIZE, y, randMax);

    cost_derivative_c_double(v, y, res_c);
    cost_derivative_asm_double(v, y, res_asm);

    assert(equalVectors_double(res_c,res_asm,SIZE));
  }

  printf("%s\n", "\t\tTests pasados exitosamente por cost_derivative");


  printf("%s\n", "\tTesteo mat_plus_vec...");

  for(uint i = 0; i < 100; i++){

    randomVector_double(SIZE, v, randMax);
    randomVector_double(SIZE, y, randMax);

    mat_plus_vec_c_double(v, y, SIZE, res_c);
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

  srand(time(NULL));

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

    //printMatrix_float(w_c_float, 1, SIZE);
    //printMatrix_float(w_asm_float, 1, SIZE);

    assert(equalVectors_float(w_c_float, w_asm_float, SIZE));

  }

  printf("%s\n", "\t\tTests pasados exitosamente por update_weight");

  printf("TODOS LOS TESTS DE FLOAT PASARON EXITOSAMENTE\n\n");


  return 0;
}

/*
  ALTAS CHANCES de que todos los valores que hay en las matrices de pesos 
  sean 0 despues del entrenamiento. Eso explicaria porque siempre se 
  obtiene el mismo resultado independientemente del input.
  Siempre se estaria obteniendo el bias de hidden to output layer.

  UPDATE: Los pesos son distintos a 0. Logre avanzar un poco, y llegue a la
  conclusión de que los pesos son tan grandes en la capa de entrada que provocan que el hidden state siempre sea 1.0 para todas las unidades. Eso 
  explicaria el por que no se depende del input. Lo que habria que ver entonces es porque no aprende pesos acordes a esto.

  Finalmente resulto ser que el codigo estaba bien. El problema en realidad 
  era que los valores con los que se inicializaban las matrices de pesos eran muy grandes (pese a ser random). Sospecho que la razon de que este problema no surgiera con la version de python es la diferencia en la calidad del sampleo. Mi solucion "artesanal" fue reescalar las valores sampleados, dividiendolos por 10.0
  Este cambio permitio replicar el accuracy expuesto por el autor del algoritmo en python

*/