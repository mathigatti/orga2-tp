#include <assert.h>
#include "tensorOps.h"
#include <unistd.h>

#define SIZE 1000
#define randMax 600
#define c 0.1

int main(){
  printf("//////// TESTEANDO ////////\n");

  double v[SIZE] = {[0 ... SIZE-1] = 0};
  double y[SIZE] = {[0 ... SIZE-1] = 0};
  double w_asm[SIZE] = {[0 ... SIZE-1] = 0};
  double w_c[SIZE] = {[0 ... SIZE-1] = 0};
  double res_c[SIZE] = {[0 ... SIZE-1] = 0};
  double res_asm[SIZE] = {[0 ... SIZE-1] = 0};

  srand(time(NULL));

  printf("%s\n", "Testeo cost_derivative...");

  for(uint i = 0; i < 100; i++){

    randomVector(SIZE, v, randMax);
    randomVector(SIZE, y, randMax);

    cost_derivative_c(v, y, res_c);
    cost_derivative_asm(v, y, res_asm);

    assert(equalVectors(res_c,res_asm,SIZE));
  }

  printf("%s\n", "Tests pasados exitosamente por cost_derivative");


  printf("%s\n", "Testeo mat_plus_vec...");

  for(uint i = 0; i < 100; i++){

    randomVector(SIZE, v, randMax);
    randomVector(SIZE, y, randMax);

    mat_plus_vec_c(v, y, SIZE, 1, res_c);
    mat_plus_vec_asm(v, y, SIZE, 1, res_asm);

    assert(equalVectors(res_c,res_asm,SIZE));

  }

  printf("%s\n", "Tests pasados exitosamente por mat_plus_vec");

  printf("%s\n", "Testeo update_weight...");

  for(uint i = 0; i < 100; i++){

    randomVector(SIZE, w_asm, randMax);
    randomVector(SIZE, y, randMax);
    vecCopy(w_c, w_asm, SIZE);
    
    update_weight_c(w_c, y, SIZE, c);
    update_weight_asm(w_asm, y, SIZE, c);

    assert(equalVectors(w_c,w_asm,SIZE));

  }

  printf("%s\n", "Tests pasados exitosamente por update_weight");

  printf("%s\n", "Testeo hadamardProduct...");

  for(uint i = 0; i < 100; i++){

    randomVector(SIZE, v, randMax);
    randomVector(SIZE, y, randMax);

    hadamardProduct_c(v, y, SIZE, 1, res_c);
    hadamardProduct_asm(v, y, SIZE, 1, res_asm);
    
    assert(equalVectors(res_c,res_asm,SIZE));

  }

  printf("%s\n", "Tests pasados exitosamente por hadamardProduct");

  printf("//////// TODOS LOS TESTS PASARON EXITOSAMENTE ////////\n");

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