#include "nn.h"
#include <time.h>

/*
The architechture of the network consist of
784 input units ---> a custom number of hidden units ---> 10 output units
in a full-connected style
*/

void initialize_net(Network* net, uint num_of_hid_units, float eta){
  srand(time(NULL));
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
  vector_sum(resProduct1, net->bias_in_to_hid, rows, z);

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
  vector_sum(resProduct2, net->bias_hid_to_out, rows, z);

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
    for(uint j = 0; j < n; j += mini_batch_size){
      update_mini_batch(net, training_data, j, j + mini_batch_size <= training_data->size ? j + mini_batch_size : training_data->size);
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

  float* dnw_in_to_hid = (float*) malloc(h * 784 * sizeof(float));    // h x 784
  float* dnb_in_to_hid = (float*) malloc(h * sizeof(float));   // h x 1
  float* dnw_hid_to_out = (float*) malloc(h * 10 * sizeof(float));     // 10 x h
  float* dnb_hid_to_out = (float*) malloc(10 * sizeof(float)); // 10 x 1

  uint cant_imgs = end - start;

  backprop(net, &minibatch->mat[start*784], cant_imgs, &minibatch->res[start], dnw_in_to_hid, dnb_in_to_hid, dnw_hid_to_out, dnb_hid_to_out);
  vector_sum(nabla_w_in_to_hid, dnw_in_to_hid, h * 784, nabla_w_in_to_hid);
  vector_sum(nabla_b_in_to_hid, dnb_in_to_hid, h, nabla_b_in_to_hid);
  vector_sum(nabla_w_hid_to_out, dnw_hid_to_out, h * 10, nabla_w_hid_to_out);
  vector_sum(nabla_b_hid_to_out, dnb_hid_to_out, 10, nabla_b_hid_to_out);

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

void backprop(Network* net, float* imgs, int cant_imgs, int* targets, float* nw_in_to_hid, float* nb_in_to_hid, float* nw_hid_to_out, float* nb_hid_to_out){
/*Return a tuple ``(nabla_b, nabla_w)`` representing the
gradient for the cost function C_x.  ``nabla_b`` and
``nabla_w`` are layer-by-layer lists of numpy arrays, similar
to ``self.biases`` and ``self.weights``.*/
  uint inputUnits = 784;
  uint outputUnits = 10;
  int h = net->num_of_hid_units;

  float* activation0 = (float*) malloc(cant_imgs * inputUnits * sizeof(float));
  transpose(imgs, cant_imgs, inputUnits, activation0);

  //--- FeedForward ---//

  // Ciclo 1

  float* resProduct1 = (float*) malloc(h * cant_imgs * sizeof(float));
  matrix_prod(net->w_in_to_hid, activation0, h, inputUnits, cant_imgs, resProduct1);

  float* z1 = (float*) malloc(h * cant_imgs * sizeof(float));
  mat_plus_vec(resProduct1, net->bias_in_to_hid, h, cant_imgs, z1);

  float* activation1 = (float*) malloc(h * cant_imgs * sizeof(float));
  sigmoid_v(z1, h, cant_imgs, activation1);

  free(resProduct1);
  // Ciclo 2

  float* resProduct2 = (float*) malloc(outputUnits * cant_imgs * sizeof(float));
  matrix_prod(net->w_hid_to_out, activation1, outputUnits, h, cant_imgs, resProduct2);

  float* z2 = (float*) malloc(outputUnits * cant_imgs * sizeof(float));
  mat_plus_vec(resProduct2, net->bias_hid_to_out, outputUnits, cant_imgs, z2);

  float* activation2 = (float*) malloc(outputUnits * cant_imgs * sizeof(float));
  sigmoid_v(z2, outputUnits, cant_imgs, activation2);

  free(resProduct2);
  //--- BackProp ---//

  // Ciclo 1

  resProduct1 = (float*) malloc(outputUnits * cant_imgs * sizeof(float));

  // Creo la matriz con los targets
  float* y = (float*) calloc(outputUnits * cant_imgs, sizeof(float));
  for (int j = 0; j < cant_imgs; j++) {
    y[targets[j] * cant_imgs + j] = 1;
  }

  // (y - t)
  cost_derivative(activation2, y, cant_imgs, resProduct1);

  resProduct2 = (float*) malloc(outputUnits * cant_imgs * sizeof(float));

  // y(1-y)
  sigmoid_prime_v(z2, outputUnits, cant_imgs, resProduct2);

  float* delta = (float*) malloc(outputUnits * cant_imgs * sizeof(float));

  // y(1-y)(y-t)
  hadamard_product(resProduct1, resProduct2, outputUnits, cant_imgs, delta);

  compress(delta, outputUnits, cant_imgs, nb_hid_to_out);

  free(y);
  free(resProduct1);
  free(resProduct2);

  // xy(1-y)(y-t)
  float* aux = (float*) malloc(cant_imgs * h * sizeof(float));
  transpose(activation1, h, cant_imgs, aux);
  matrix_prod(delta, aux, outputUnits, cant_imgs, h, nw_hid_to_out);

  free(aux);

  // Ciclo 2

  resProduct1 = (float*) malloc(h * cant_imgs * sizeof(float));

  sigmoid_prime_v(z1, h, cant_imgs, resProduct1);

  resProduct2 = (float*) malloc(h * cant_imgs * sizeof(float));

  // Hay que transponer net->w_hid_to_out
  aux = (float*) malloc(h * outputUnits * sizeof(float));
  transpose(net->w_hid_to_out, outputUnits, h, aux);
  matrix_prod(aux, delta, h, outputUnits, cant_imgs, resProduct2);

  free(aux);
  free(delta);

  delta = (float*) malloc(h * cant_imgs * sizeof(float));
  hadamard_product(resProduct1, resProduct2, h, cant_imgs, delta);

  compress(delta, h, cant_imgs, nb_in_to_hid);

  free(resProduct1);
  free(resProduct2);

  aux = (float*) malloc(cant_imgs * inputUnits * sizeof(float));
  transpose(activation0, inputUnits, cant_imgs, aux);
  matrix_prod(delta, aux, h, cant_imgs, inputUnits, nw_in_to_hid);

  //--- Libero memoria ---//

  free(aux);
  free(delta);
  free(z1);
  free(z2);

  free(activation0);
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

//TOFIX: remove it
double calculateMean(double data[])
{
    double sum = 0.0;

    for(int i=0; i<ITERACIONES; i++) {
        sum += data[i];
    }

    return sum/ITERACIONES;
}

double calculateSD(double data[])
{
    double standardDeviation = 0.0;
    double mean = calculateMean(data);
    for(int i=0; i<ITERACIONES; i++)
        standardDeviation += pow(data[i] - mean, 2);

    return sqrt(standardDeviation/ITERACIONES);
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

  struct timespec tstart, tend;
  clock_gettime(CLOCK_MONOTONIC, &tstart);
  SGD(net, training_data, EPOCHS, MINI_BATCH_SIZE, net->eta);
  clock_gettime(CLOCK_MONOTONIC, &tend);
  printf("Average time per epoch: %f\n", ((tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1000000000.0) / EPOCHS);

  feed_forward(net, &test_data->mat[1 * 784], 1, res);

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

  float v[MINI_BATCH_SIZE * 10];
  float w[MINI_BATCH_SIZE * 10];
  float u[MINI_BATCH_SIZE * 10];

  uint n = 10;
  uint m = 20;
  uint l = 30;

  float m1[n*m];
  float m2[m*l];
  float m_res[n*l];

  srand(time(NULL));

  double times[ITERACIONES];
  for(int i = 0; i < ITERACIONES; i++){
    randomMatrix(v, MINI_BATCH_SIZE, 10);
    randomMatrix(w, MINI_BATCH_SIZE, 10);
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    cost_derivative(v, w, MINI_BATCH_SIZE, u);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    times[i] = (tend.tv_sec - tstart.tv_sec) * 1000000.0 + (tend.tv_nsec - tstart.tv_nsec) / 1000.0;
  }

  printf("Average time cost_derivative (in milli sec): %f\n", calculateMean(times));
  printf("Standard deviation for cost_derivative: %f\n", calculateSD(times));
  
  for(int i = 0; i < ITERACIONES; i++){
    randomVector(SIZE, v, randMax);
    randomVector(SIZE, w, randMax);
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    vector_sum(v, w, SIZE, u);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    times[i] = (tend.tv_sec - tstart.tv_sec) * 1000000.0 + (tend.tv_nsec - tstart.tv_nsec) / 1000.0;
  }

  printf("Average time vector_sum (in milli sec): %f\n", calculateMean(times));
  printf("Standard deviation for vector_sum: %f\n", calculateSD(times));

  for(int i = 0; i < ITERACIONES; i++){
    randomVector(SIZE, v, randMax);
    randomVector(SIZE, w, randMax);
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    update_weight(v, w, SIZE, 0.3);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    times[i] = (tend.tv_sec - tstart.tv_sec) * 1000000.0 + (tend.tv_nsec - tstart.tv_nsec) / 1000.0;
  }

  printf("Average time update_weight (in milli sec): %f\n", calculateMean(times));
  printf("Standard deviation for update_weight: %f\n", calculateSD(times));

  for(int i = 0; i < ITERACIONES; i++){
    randomVector(MINI_BATCH_SIZE * 10, v, randMax);
    randomVector(MINI_BATCH_SIZE * 10, w, randMax);
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    hadamard_product(v, w, 10, MINI_BATCH_SIZE, u);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    times[i] = (tend.tv_sec - tstart.tv_sec) * 1000000.0 + (tend.tv_nsec - tstart.tv_nsec) / 1000.0;
  }

  printf("Average time hadamardProduct (in milli seconds): %f\n", calculateMean(times));
  printf("Standard deviation for hadamard_product: %f\n", calculateSD(times));

  for(int i = 0; i < ITERACIONES; i++){
    randomMatrix(m1, n, m);
    randomMatrix(m2, m, l);
    struct timespec tstart, tend;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    matrix_prod(m1, m2, n, m, l, m_res);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    times[i] = (tend.tv_sec - tstart.tv_sec) * 1000000.0 + (tend.tv_nsec - tstart.tv_nsec) / 1000.0;
  }

  printf("Average time matrix_prod (in milli seconds): %f\n", calculateMean(times));
  printf("Standard deviation for matrix_prod: %f\n", calculateSD(times));

  return 0;
}