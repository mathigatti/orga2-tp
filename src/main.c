#include <stdio.h>
#include <stdlib.h>

typedef unsigned int uint;

double* training_data; 
double* test_data;
double* training_labels;
double* test_labels;

typedef struct Network {
  /*The list ``sizes`` contains the number of neurons in the
    respective layers of the network.  For example, if the list
    was [2, 3, 1] then it would be a three-layer network, with the
    first layer containing 2 neurons, the second layer 3 neurons,
    and the third layer 1 neuron.  The biases and weights for the
    network are initialized randomly, using a Gaussian
    distribution with mean 0, and variance 1.  Note that the first
    layer is assumed to be an input layer, and by convention we
    won't set any biases for those neurons, since biases are only
    ever used in computing the outputs from later layers.*/
  uint num_layers;
  uint* sizes;
  double* biases;
  double* weights;
} Network;

double* feed_forward(Network* net, double* input); 
/*Return the output of the network if ``a`` is input.*/

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

double sigmoid(double z);
/*The sigmoid function.*/

double sigmoid_prime(double z);
/*Derivative of the sigmoid function.*/

int main (void){
  Network net;
  int sizes[3] = {17,72,73};
  net.sizes = sizes;

  for(int i = 0; i < 3; i++)
    printf("El tamaño %d es %d \n", i+1, net.sizes[i]);

  return 0;
}