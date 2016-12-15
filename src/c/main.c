#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nn.h"

double* training_data; 
double* test_data;
double* training_labels;
double* test_labels;

/*The architechture of the network consist of
784 input units ---> a custom number of hidden units ---> 10 output units
*/

void initialize_net(Network* net, uint num_of_hid_units){
  //srand(time(NULL));
  net->num_of_hid_units = num_of_hid_units;
  net->bias_in_to_hid = (double*) malloc(num_of_hid_units*sizeof(double));
  net->bias_hid_to_out = (double*) malloc(num_of_hid_units*sizeof(double));
  net->w_in_to_hid = (double*) malloc(784 * num_of_hid_units * sizeof(double));
  net->w_hid_to_out = (double*) malloc(10 * num_of_hid_units * sizeof(double));
  
  for(int i = 0; i < num_of_hid_units; i++) { 
    net->bias_in_to_hid[i] = (double) rand() / RAND_MAX;
    for(int j = 0; j < 784; j++)
      net->w_in_to_hid[i * num_of_hid_units + j] = (double) rand() / RAND_MAX;     
  }

  for(int i = 0; i < 10; i++){
    net->bias_hid_to_out[i] = (double) rand() / RAND_MAX;
    for(int j = 0; j < num_of_hid_units; j++)
      net->w_hid_to_out[i * 10 + j] = (double) rand() / RAND_MAX;
  }
}

double* feed_forward(Network* net, double* input) {
/*Return the output of the network if ``a`` is input.*/
  uint rows = net->num_of_hid_units;
  uint cols = 784;
  double* z = sum_vec(matrix_vec_prod(net->w_in_to_hid, input, rows, cols), net->bias_in_to_hid, rows);
  double* hidden_state = sigmoid_v(z, rows);

  rows = 10;
  cols = net->num_of_hid_units;
  z = sum_vec(matrix_vec_prod(net->w_hid_to_out, hidden_state, rows, cols), net->bias_hid_to_out, rows);

  return sigmoid_v(z, rows);
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

double* backprop(Network* net, double* X, double* y);
/*Return a tuple ``(nabla_b, nabla_w)`` representing the
gradient for the cost function C_x.  ``nabla_b`` and
``nabla_w`` are layer-by-layer lists of numpy arrays, similar
to ``self.biases`` and ``self.weights``.*/

int evaluate(Network* net, double* test_data);
/*Return the number of test inputs for which the neural
  network outputs the correct result. Note that the neural
  network's output is assumed to be the index of whichever
  neuron in the final layer has the highest activation.*/

double* cost_derivative(double* output_activations, double* y);
/*Return the vector of partial derivatives \partial C_x /
  \partial a for the output activations.*/

double sigmoid(double z){
/*The sigmoid function.*/
  return 1/(1 + exp(-z));
}

double* sigmoid_v(double* z, uint n){
/*The sigmoid function.*/
  double* y = (double*) malloc(n * sizeof(double));
  for(uint i = 0; i < n; i++) {
    y[i] = sigmoid(z[i]);
  }
  return y;
}

double sigmoid_prime(double z);
/*Derivative of the sigmoid function.*/

double* sum_vec(double* v, double* w, uint n){
/*Sum of vectors*/
/*TO OPTIMIZE*/
  double* sum = (double*) malloc(n * sizeof(double));
  for(uint i = 0; i < n; i++){
    sum[i] = v[i] + w[i];
  }
  return sum;
}

double* matrix_vec_prod(double* W, double* x, uint rows, uint cols){
/*Usual matrix product*/
/*TO OPTIMIZE*/
  double* y = (double*) malloc(sizeof(double) * rows);
  for(uint i = 0; i < rows; i++){
    y[i] = 0;
    for(uint j = 0; j < cols; j++){
      y[i*cols+j] += W[i*cols + j] * x[j];
    }
  }
  return y;
}

int main(){
  Network* net = (Network*) malloc(sizeof(Network));
  initialize_net(net, 1);

  printf("La cantidad de unidades ocultas es %d \n", net->num_of_hid_units);
  double input[784] = {[0 ... 783] = 0};
  double* y = feed_forward(net, input);
  for(int i = 0; i < 10; i++){
    printf("Valor asignado a %d: %f\n", i, y[i]);
  }
  free(net);
  return 0;
}