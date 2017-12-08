#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensorOps.h"

#define MINI_BATCH_SIZE 32
#define SIZE 1000
#define randMax 10
#define ITERACIONES 400000
#define EPOCHS 8

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

void SGD(Network* net, Images* training_data, uint epochs, uint mini_batch_size, double eta);

void update_mini_batch(Network* net, Images* minibatch, uint start, uint end);

void backprop(Network* net, double* imgs, int cant_imgs, int* target, double* nw_in_to_hid, double* nb_in_to_hid, double* nw_hid_to_out, double* nb_hid_to_out);

double evaluate(Network* net, Images* test_data);

void predictNumber(Network* net, const char* txtImage);

void calculateTime();