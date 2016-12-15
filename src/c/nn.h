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
} Network;

void initialize_net(Network* net, uint num_of_hid_units);

double* feed_forward(Network* net, double* input);

void SGD(Network* net, double* training_data, uint epochs, uint mini_batch_size, double eta);

void update_mini_batch(Network* net);

double* backprop(Network* net, double* X, double* y);

int evaluate(Network* net, double* test_data);

double* cost_derivative(double* output_activations, double* y);

double sigmoid(double z);

double* sigmoid_v(double* z, uint n);

double sigmoid_prime(double z);

double* sum_vec(double* v, double* w, uint n);

double* matrix_vec_prod(double* W, double* x, uint rows, uint cols);