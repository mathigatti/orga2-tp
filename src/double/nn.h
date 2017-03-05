#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensorOps.h"

#define MINI_BATCH_SIZE 10

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

void backprop(Network* net, double* X, int target, double* , double*, double*, double*);

double evaluate(Network* net, Images* test_data);

// Para arreglar:

/*

Los errores que esta tirando valgrind (conditional jump...) se deben 
a algun problema con la inicializacion de z1 y z2 en backpropagation. 
Efectivamente el error se soluciona usando calloc en vez de malloc, pero en teoria esto no deberia ser necesario porque se supone que SI estamos inicializando los valores.

*/

// Ideas en general

/*

Probar regularizar pesos de la red

Testeo automatico de implementaciones en ASM con la version en C


*/


// Ideas para experimentacion

/* 

Experimentar performance temporal funcion por funcion
Probar variante en funciones, loop unrolling
Multithreading

*/