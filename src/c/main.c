#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nn.h"
#include "txtReader.c"

/*The architechture of the network consist of
784 input units ---> a custom number of hidden units ---> 10 output units
*/

void initialize_net(Network* net, uint num_of_hid_units){
  srand(time(NULL));
  net->num_of_hid_units = num_of_hid_units;
  net->bias_in_to_hid = (double*) malloc(num_of_hid_units*sizeof(double));
  net->bias_hid_to_out = (double*) malloc(num_of_hid_units*sizeof(double));
  net->w_in_to_hid = (double*) malloc(784 * num_of_hid_units * sizeof(double));
  net->w_hid_to_out = (double*) malloc(10 * num_of_hid_units * sizeof(double));
  
  for(int i = 0; i < num_of_hid_units; i++) { 
    net->bias_in_to_hid[i] = (double) rand() / RAND_MAX;
    for(int j = 0; j < 784; j++) {
      net->w_in_to_hid[i * num_of_hid_units + j] = (double) rand() / RAND_MAX;     
    }
  }

  for(int i = 0; i < 10; i++){
    net->bias_hid_to_out[i] = (double) rand() / RAND_MAX;
    for(int j = 0; j < num_of_hid_units; j++)
      net->w_hid_to_out[i * 10 + j] = (double) rand() / RAND_MAX;
  }
}

void destructor_net(Network* net) {
  free(net->bias_in_to_hid);
  free(net->bias_hid_to_out);
  free(net->w_in_to_hid);
  free(net->w_hid_to_out);
  free(net);
}

void feed_forward(Network* net, double* input, uint cant_img, double* output) {
/*Return the output of the network if ``a`` is input.*/
  uint rows = net->num_of_hid_units;
  uint cols = 784;

  double* resProduct1 = (double*) malloc(rows * cant_img * sizeof(double));
  matrix_vec_prod(net->w_in_to_hid, input, rows, cols, cant_img, resProduct1);

  double* z = (double*) malloc(rows * cant_img * sizeof(double));
  sum_vec(resProduct1, net->bias_in_to_hid, rows, cant_img, z);

  double* hidden_state = (double*) malloc(rows * cant_img * sizeof(double));

  sigmoid_v(z, rows, cant_img, hidden_state);

  free(z);
  free(resProduct1);

  rows = 10;
  cols = net->num_of_hid_units;

  double* resProduct2 = (double*) malloc(rows * cant_img * sizeof(double));

  z = (double*) malloc(rows * cant_img * sizeof(double));

  matrix_vec_prod(net->w_hid_to_out, hidden_state, rows, cols, cant_img, resProduct2);
  sum_vec(resProduct2, net->bias_hid_to_out, rows, cant_img, z);

  sigmoid_v(z, rows, cant_img, output);
 
  free(resProduct2);
  free(hidden_state);
  free(z);
}

void SGD(Network* net, double* training_data, uint epochs, uint mini_batch_size, double eta);
/*Train the neural network using mini-batch stochastic
  gradient descent.  The ``training_data`` is a list of tuples
  ``(x, y)`` representing the training inputs and the desired
  outputs.  The other non-optional parameters are
  self-explanatory.  If ``test_data`` is provided then the
  network will be evaluated against the test data after each
  epoch, and partial progress printed out.  This is useful for
  tracking progress, but slows things down substantially.*/

void update_mini_batch(Network* net);
/*Update the network's weights and biases by applying
  gradient descent using backpropagation to a single mini batch.
  The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
  is the learning rate.*/

void backprop(Network* net, double* X, double* y, double* output);
/*Return a tuple ``(nabla_b, nabla_w)`` representing the
gradient for the cost function C_x.  ``nabla_b`` and
``nabla_w`` are layer-by-layer lists of numpy arrays, similar
to ``self.biases`` and ``self.weights``.*/

int evaluate(Network* net, double* test_data);
/*Return the number of test inputs for which the neural
  network outputs the correct result. Note that the neural
  network's output is assumed to be the index of whichever
  neuron in the final layer has the highest activation.*/

void cost_derivative(double* output_activations, double* y, uint n, double* output) {
/*Return the vector of partial derivatives \partial C_x /
  \partial a for the output activations.*/
  for(uint i = 0; i < n; i++){
    output[i] = output_activations[i] - y[i];
  }
}

double sigmoid(double z){
/*The sigmoid function.*/
  return 1/(1 + exp(-z));
}

void sigmoid_v(double* z, uint n, uint cant_img, double* output){
/*The sigmoid function.*/
  for(uint k = 0; k < cant_img; k++){
    for(uint i = 0; i < n; i++) {
      output[k * n + i] = sigmoid(z[k * n + i]);
    }
  }
}

double sigmoid_prime(double z){
/*The sigmoid function.*/
  return sigmoid(z)*(1-sigmoid(z));
}


void sigmoid_prime_v(double* z, uint n, double* output){
  for(uint i = 0; i < n; i++){
    double sigmoidea = sigmoid(z[i]);
    double minusOneSigmoid = 1 - sigmoidea;
    output[i] = minusOneSigmoid*sigmoidea;
  }
}

void sum_vec(double* v, double* w, uint n, uint cant_img, double* output){
/*Sum of vectors*/
/*TO OPTIMIZE*/
  for(int k = 0; k < cant_img; k++){
    for(uint i = 0; i < n; i++){
      output[k * n + i] = v[k * n + i] + w[i];
    }
  }
}

void matrix_vec_prod(double* W, double* X, uint rows, uint cols, uint cant_img, double* output){
/*Usual matrix product*/
/*TO OPTIMIZE*/
  for(uint k = 0; k < cant_img; k++) {
    for(uint i = 0; i < rows; i++){
      output[i * cant_img + k] = 0;
      for(uint j = 0; j < cols; j++){
        output[i * cant_img + k] += W[i*cols + j] * X[j * cant_img + k];
      }
    }
  }
}

void shuffle(int *array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int main(){
  Imagenes* training_data = trainSetReader();
  Imagenes* test_data = testSetReader();
  Network* net = (Network*) malloc(sizeof(Network));
  initialize_net(net, 10);

  //testeo feedforward con un mini-batch
  double* res = (double*) malloc(10 * MINI_BATCH_SIZE * sizeof(double));

  feed_forward(net, training_data->mat_tr, MINI_BATCH_SIZE, res);
  for(int i = 0; i < 10; i++){
    printf("Valor para %d: %f\n", i, res[i]);
  }

  free(res);
  
  //testeo feedforward con todos 0's
  double input[784] = {[0 ... 783] = 0};

  double* y = (double*) malloc(10 * sizeof(double));

  feed_forward(net, input, 1, y);
  for(int i = 0; i < 10; i++){
    printf("Valor asignado a %d: %f\n", i, y[i]);
  }

  free(y);
  destructor_net(net);
  free(training_data);
  free(test_data);

  return 0;
}