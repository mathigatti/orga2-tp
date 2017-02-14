#include "txtReader.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MINI_BATCH_SIZE 100

typedef unsigned int uint;

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
  uint num_of_hid_units;    // h = #hidden units
  double* bias_in_to_hid;   // h x 1
  double* bias_hid_to_out;  // 10 x 1
  double* w_in_to_hid;      // h x 784
  double* w_hid_to_out;     // 10 x h
  double eta; //learning rate
} Network;

void initialize_net(Network* net, uint num_of_hid_units, double eta);

void destructor_net(Network* net);

void feed_forward(Network* net, double* input, uint cant_img, double* output);

void SGD(Network* net, Imagenes* training_data, uint epochs, uint mini_batch_size, double eta);

void update_mini_batch(Network* net, Imagenes* minibatch, uint start, uint end);

void update_weight(double* w, double* nw, uint w_size, uint mb_size, double eta);

void backprop(Network* net, double* X, int target, double* nb_hid_to_out , double* nw_hid_to_out, double* nb_in_to_hid, double* nw_in_to_hid);

int evaluate(Network* net, double* test_data);

void productoHadamard(double* matrix_1, double* matrix_2, uint rows, uint cols, double* output);

void cost_derivative(double* output_activations, double* y, uint n, uint cant_img, double* output);

double sigmoid(double z);

void sigmoid_v(double* matrix, uint rows, uint cols, double* output);

double sigmoid_prime(double z);

void sigmoid_prime_v(double* matrix, uint rows, uint cols, double* output);

void random_shuffle(Imagenes* batch);

void sum_vec(double* matrix, double* vector, uint vector_size, uint matrix_cols, double* output);

void matrix_prod(double* matrix_1, double* matrix_2, uint matrix_1_rows, uint matrix_1_cols, uint matrix_2_cols, double* output);
