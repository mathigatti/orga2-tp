#include "nn.h"
#include <time.h>

#define SIZE 5000
#define randMax 10
#define ITERACIONES 50000

/*
The architechture of the network consist of
784 input units ---> a custom number of hidden units ---> 10 output units
in a full-connected style
*/

void initialize_net(Network* net, uint num_of_hid_units, float eta){
  srand(time(NULL));
  /*
  for(uint i = 0; i < 1000; i++){
    printf("%f\n", (float) rand() / RAND_MAX);
  }
  */
  net->eta = eta;
  net->num_of_hid_units = num_of_hid_units;
  net->bias_in_to_hid = (float*) malloc(num_of_hid_units*sizeof(float));
  net->bias_hid_to_out = (float*) malloc(10*sizeof(float));
  net->w_in_to_hid = (float*) malloc(784 * num_of_hid_units * sizeof(float));
  net->w_hid_to_out = (float*) malloc(10 * num_of_hid_units * sizeof(float));
  
  for(int i = 0; i < num_of_hid_units; i++) { 
    net->bias_in_to_hid[i] = (float) rand() / RAND_MAX;
    net->bias_in_to_hid[i] /= 10.0;
    for(int j = 0; j < 784; j++) {
      net->w_in_to_hid[i * 784 + j] = (float) rand() / RAND_MAX;     
      net->w_in_to_hid[i * 784 + j] /= 10.0;     
    }
  }

  for(int i = 0; i < 10; i++){
    net->bias_hid_to_out[i] = (float) rand() / RAND_MAX;
    net->bias_hid_to_out[i] /= 10.0;
    for(int j = 0; j < num_of_hid_units; j++) {
      net->w_hid_to_out[i * num_of_hid_units + j] = (float) rand() / RAND_MAX;
      net->w_hid_to_out[i * num_of_hid_units + j] /= 10.0;
    }
  }
}

void destructor_net(Network* net) {
  free(net->bias_in_to_hid);
  free(net->bias_hid_to_out);
  free(net->w_in_to_hid);
  free(net->w_hid_to_out);
  free(net);
}

void feed_forward(Network* net, float* input, uint cant_img, float* output) {
/*Return the output of the network if ``a`` is input.*/
  float* input_tr = (float*) malloc(784 * cant_img * sizeof(float));
  transpose(input, cant_img, 784, input_tr);

  uint rows = net->num_of_hid_units;
  uint cols = 784;
  cant_img = 1;

  float* resProduct1 = (float*) malloc(rows * cant_img * sizeof(float));

  matrix_prod(net->w_in_to_hid, input_tr, rows, cols, cant_img, resProduct1);

  float* z = (float*) malloc(rows * cant_img * sizeof(float));
  mat_plus_vec(resProduct1, net->bias_in_to_hid, rows, z);

  float* hidden_state = (float*) malloc(rows * cant_img * sizeof(float));

  sigmoid_v(z, rows, cant_img, hidden_state);

  free(input_tr);
  free(z);
  free(resProduct1);

  rows = 10;
  cols = net->num_of_hid_units;

  float* resProduct2 = (float*) malloc(rows * cant_img * sizeof(float));

  z = (float*) malloc(rows * cant_img * sizeof(float));

  matrix_prod(net->w_hid_to_out, hidden_state, rows, cols, cant_img, resProduct2);
  mat_plus_vec(resProduct2, net->bias_hid_to_out, rows, z);

  sigmoid_v(z, rows, cant_img, output);
 
  free(resProduct2);
  free(hidden_state);
  free(z);
}

void SGD(Network* net, Images* training_data, uint epochs, uint mini_batch_size, float eta){
/*Train the neural network using mini-batch stochastic
  gradient descent.  The ``training_data`` is a list of tuples
  ``(x, y)`` representing the training inputs and the desired
  outputs.  The other non-optional parameters are
  self-explanatory.  If ``test_data`` is provided then the
  network will be evaluated against the test data after each
  epoch, and partial progress printed out.  This is useful for
  tracking progress, but slows things down substantially.*/
  printf("Starting SGD...\n");
  uint n = IMGS_NUM;
  printf("Cantidad de imagenes: %d\n", n);

  for(uint i = 0; i < epochs; i++){
    printf("Epoch: %d\n", i);
    random_shuffle(training_data);
    //printf("Shuffle exitoso\n");
    for(uint j = 0; j < n; j += mini_batch_size){
      update_mini_batch(net, training_data, j, j + mini_batch_size);
    }
  }
  printf("SGD ended successfully\n");
}

void update_mini_batch(Network* net, Images* minibatch, uint start, uint end) {
/*Update the network's weights and biases by applying
  gradient descent using backpropagation to a single mini batch.
  The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
  is the learning rate.*/
  uint h = net->num_of_hid_units;

  float* nabla_w_in_to_hid = (float*) calloc(h * 784, sizeof(float));      // h x 784
  float* nabla_b_in_to_hid = (float*) calloc(h, sizeof(float));   // h x 1
  float* nabla_w_hid_to_out = (float*) calloc(h * 10, sizeof(float));     // 10 x h
  float* nabla_b_hid_to_out = (float*) calloc(10, sizeof(float)); // 10 x 1

  // dnb = delta nabla b
  // dnw = delta nabla w
  float* dnw_in_to_hid = (float*) malloc(h * 784 * sizeof(float));    // h x 784
  float* dnb_in_to_hid = (float*) malloc(h * sizeof(float));   // h x 1
  float* dnw_hid_to_out = (float*) malloc(h * 10 * sizeof(float));     // 10 x h
  float* dnb_hid_to_out = (float*) malloc(10 * sizeof(float)); // 10 x 1
  
  for(uint i = start; i < end; i++){
    backprop(net, &minibatch->mat[i*784], minibatch->res[i], dnw_in_to_hid, dnb_in_to_hid, dnw_hid_to_out, dnb_hid_to_out);
    mat_plus_vec(nabla_w_in_to_hid, dnw_in_to_hid, h * 784, nabla_w_in_to_hid);
    mat_plus_vec(nabla_b_in_to_hid, dnb_in_to_hid, h, nabla_b_in_to_hid);
    mat_plus_vec(nabla_w_hid_to_out, dnw_hid_to_out, h * 10, nabla_w_hid_to_out);
    mat_plus_vec(nabla_b_hid_to_out, dnb_hid_to_out, 10, nabla_b_hid_to_out);
  }

  // Free memory for delta nablas
  free(dnw_in_to_hid);
  free(dnb_in_to_hid);
  free(dnw_hid_to_out);
  free(dnb_hid_to_out);

  float c = net->eta / MINI_BATCH_SIZE; //Esto se puede calcular una sola vez para toda la red
  update_weight(net->w_in_to_hid, nabla_w_in_to_hid, h * 784, c);
  update_weight(net->bias_in_to_hid, nabla_b_in_to_hid, h, c);
  update_weight(net->w_hid_to_out, nabla_w_hid_to_out, 10 * h, c);
  update_weight(net->bias_hid_to_out, nabla_b_hid_to_out, 10, c);

  // Free memory for nablas
  free(nabla_w_in_to_hid);
  free(nabla_b_in_to_hid);
  free(nabla_w_hid_to_out);
  free(nabla_b_hid_to_out);  
}

void backprop(Network* net, float* input, int target, float* nw_in_to_hid, float* nb_in_to_hid, float* nw_hid_to_out, float* nb_hid_to_out){
/*Return a tuple ``(nabla_b, nabla_w)`` representing the
gradient for the cost function C_x.  ``nabla_b`` and
``nabla_w`` are layer-by-layer lists of numpy arrays, similar
to ``self.biases`` and ``self.weights``.*/

  int h = net->num_of_hid_units;

  float* activation0 = input;

  //--- FeedForward ---//

  // Ciclo 1

  uint cant_img = 1;
  uint inputUnits = 784;
  uint outputUnits = 10;

  float* resProduct1 = (float*) malloc(h * cant_img * sizeof(float));

  matrix_prod(net->w_in_to_hid, activation0, h, inputUnits, cant_img, resProduct1);

  float* z1 = (float*) malloc(h * cant_img * sizeof(float));
  mat_plus_vec(resProduct1, net->bias_in_to_hid, h, z1);

  float* activation1 = (float*) malloc(h * cant_img * sizeof(float));
  sigmoid_v(z1, h, cant_img, activation1);

  free(resProduct1);

  // Ciclo 2

  float* resProduct2 = (float*) malloc(outputUnits * cant_img * sizeof(float));
  matrix_prod(net->w_hid_to_out, activation1, outputUnits, h, cant_img, resProduct2);

  float* z2 = (float*) malloc(outputUnits * cant_img * sizeof(float));
  mat_plus_vec(resProduct2, net->bias_hid_to_out, outputUnits, z2);

  float* activation2 = (float*) malloc(outputUnits * cant_img * sizeof(float));
  sigmoid_v(z2, outputUnits, cant_img, activation2);  

  free(resProduct2);

  //--- BackProp ---//

  // Ciclo 1

  resProduct1 = (float*) malloc(outputUnits * cant_img * sizeof(float));

  float y[10] = {[0 ... 9] = 0};
  y[target] = 1;

  // (y - t)
  cost_derivative(activation2, y, resProduct1); 

  resProduct2 = (float*) malloc(outputUnits * cant_img * sizeof(float));

  // y(1-y)
  sigmoid_prime_v(z2, outputUnits, 1, resProduct2);

  // y(1-y)(y-t)
  hadamardProduct(resProduct1, resProduct2, outputUnits, 1, nb_hid_to_out);

  free(resProduct1);  
  free(resProduct2);

  // xy(1-y)(y-t)
  matrix_prod(nb_hid_to_out, activation1, outputUnits, 1, h, nw_hid_to_out);

  // Hasta aca me cierra bien
  // Ciclo 2

  resProduct1 = (float*) malloc(h * cant_img * sizeof(float));

  sigmoid_prime_v(z1, h, 1, resProduct1);

  resProduct2 = (float*) malloc(h * cant_img * sizeof(float));

  // Hay que transponer net->w_hid_to_out 
  float* aux = (float*) malloc(h * 10 * sizeof(float));
  transpose(net->w_hid_to_out, 10, h, aux);
  matrix_prod(aux, nb_hid_to_out, h, 10, 1, resProduct2);
  free(aux);

  hadamardProduct(resProduct1, resProduct2, h, 1, nb_in_to_hid);

  free(resProduct1);  
  free(resProduct2);

  matrix_prod(nb_in_to_hid, activation0, h, 1, inputUnits, nw_in_to_hid);

  //--- Libero memoria ---//

  free(z1);
  free(z2);

  free(activation1);
  free(activation2);

}

float evaluate(Network* net, Images* test_data){
/*Return the accuracy over test_data*/
  int hits = 0;
  float* res = (float*) malloc(10 * sizeof(float));

  for(uint i = 0; i < test_data->size; i++) {
    feed_forward(net, &test_data->mat[i * 784], 1, res);
    int y = max_arg(res, 10);
    if(y == test_data->res[i]) {
      hits++;
    }
  }
  free(res);

  return hits / (float) test_data->size;
}

int main(){
  Images* training_data = trainSetReader();
  Images* test_data = testSetReader();
  Network* net = (Network*) malloc(sizeof(Network));
  initialize_net(net, 30, 3.0);

  if (training_data == NULL || test_data == NULL){
    printf("Error intentando leer data-sets de entrada\n");
    return 0;
  }

  //testeo SGD con un mini-batch
  float* res = (float*) malloc(10 * sizeof(float));
  
  SGD(net, training_data, 4, MINI_BATCH_SIZE, net->eta);

  feed_forward(net, &test_data->mat[1 * 784], 1, res);
  // printImg(&test_data->mat[1 * 784]);
  // printf("Target: %d\n", test_data->res[3]);
  for(int i = 0; i < 10; i++){
    printf("Valor para %d: %f\n", i, res[i]);
  }
  printf("Target: %d\n", test_data->res[1]);

  //testeo feedforward con un 1 artificial
  float input[784] = {[0 ... 783] = 0};
  for(uint i = 42; i < 784; i += 28){
    input[i] = 1.0;
  }

  feed_forward(net, input, 1, res);
  for(int i = 0; i < 10; i++){
    printf("Valor asignado a %d: %f\n", i, res[i]);
  }
  printf("Target: %d\n", 1);

  free(res);

  // Evaluate accuracy over data set
  printf("Accuracy over testing data set: %f\n", evaluate(net, test_data));

  destructor_net(net);
  imagesDestructor(training_data);
  imagesDestructor(test_data);
  /*
  clock_t start, end;
  float cpu_time_used = 0;
  

  float v[SIZE];
  float w[SIZE];
  float u[SIZE];

  uint n = 10;
  uint m = 20;
  uint l = 30;

  float m1[n*m];
  float m2[m*l];
  float m_res[n*l];

  srand(time(NULL));

  for(int i = 0; i < ITERACIONES; i++){
    randomVector(SIZE, v, randMax);
    randomVector(SIZE, w, randMax);
    start = clock();
    cost_derivative(v, w, u);
    end = clock();
    cpu_time_used += ((float) end - start) / CLOCKS_PER_SEC;
  }
  //cpu_time_used /= 1000000.0;
  printf("Total time cost_derivative: %f\n", cpu_time_used);


  cpu_time_used = 0;
  for(int i = 0; i < ITERACIONES; i++){
    randomVector(SIZE, v, randMax);
    randomVector(SIZE, w, randMax);
    start = clock();
    mat_plus_vec(v, w, SIZE, u);
    end = clock();
    cpu_time_used += ((float) end - start) / CLOCKS_PER_SEC;
  }
  //cpu_time_used /= 1000000.0;
  printf("Total time mat_plus_vec: %f\n", cpu_time_used);

  cpu_time_used = 0;
  for(int i = 0; i < ITERACIONES; i++){
    randomVector(SIZE, v, randMax);
    randomVector(SIZE, w, randMax);
    start = clock();
    update_weight(v, w, SIZE, 0.3);
    end = clock();
    cpu_time_used += ((float) end - start) / CLOCKS_PER_SEC;
  }
  //cpu_time_used /= 1000000.0;
  printf("Total time update_weight: %f\n", cpu_time_used);

  cpu_time_used = 0;
  for(int i = 0; i < ITERACIONES; i++){
    randomVector(SIZE, v, randMax);
    randomVector(SIZE, w, randMax);
    start = clock();
    hadamardProduct(v, w, SIZE, 1, u);
    end = clock();
    cpu_time_used += ((float) end - start) / CLOCKS_PER_SEC;
  }
  //cpu_time_used /= 1000000.0;
  printf("Total time hadamardProduct: %f\n", cpu_time_used);

  cpu_time_used = 0;
  for(int i = 0; i < ITERACIONES; i++){
    randomMatrix(m1, n, m);
    randomMatrix(m2, m, l);
    start = clock();
    matrix_prod(m1, m2, n, m, l, m_res);
    end = clock();
    cpu_time_used += ((float) end - start) / CLOCKS_PER_SEC;
  }
  //cpu_time_used /= 1000000.0;
  printf("Total time matrix_prod: %f\n", cpu_time_used);
  */
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