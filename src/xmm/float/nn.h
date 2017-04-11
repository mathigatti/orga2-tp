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
  float* bias_in_to_hid;   // h x 1
  float* bias_hid_to_out;  // 10 x 1
  float* w_in_to_hid;      // h x 784
  float* w_hid_to_out;     // 10 x h
  float eta; //learning rate
} Network;

void initialize_net(Network* net, uint num_of_hid_units, float eta);

void destructor_net(Network* net);

void feed_forward(Network* net, float* input, uint cant_img, float* output);

void SGD(Network* net, Images* training_data, uint epochs, uint mini_batch_size, float eta);

void update_mini_batch(Network* net, Images* minibatch, uint start, uint end);

void backprop(Network* net, float* X, int target, float* , float*, float*, float*);

float evaluate(Network* net, Images* test_data);

// Para arreglar:

/* Nada por ahora :) */

// Ideas en general

/*

Probar regularizar pesos de la red

Testeo automatico de implementaciones en ASM con la version en C

Fijarnos si no tenemos disponible AVX (registros ymm en vez de xmm). Aparentemente por lo que lei deberiamos tenerlos.


*/


// Ideas para experimentacion

/* 

Experimentar performance temporal funcion por funcion
Probar variante en funciones, loop unrolling
Multithreading
Ver si hay ciertos parametros para los cuales la optimizacion
con SIMD es aun mayor (por ej, numeros multiplos de 4)

*/