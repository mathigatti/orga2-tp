#include "nn.h"

/*The architechture of the network consist of
784 input units ---> a custom number of hidden units ---> 10 output units
*/

void initialize_net(Network* net, uint num_of_hid_units, double eta){
  srand(time(NULL));
  net->eta = eta;
  net->num_of_hid_units = num_of_hid_units;
  net->b_in_to_hid = (double*) malloc(num_of_hid_units*sizeof(double));
  net->b_hid_to_out = (double*) malloc(10*sizeof(double));
  net->w_in_to_hid = (double*) malloc(784 * num_of_hid_units * sizeof(double));
  net->w_hid_to_out = (double*) malloc(10 * num_of_hid_units * sizeof(double));
  
  for(int i = 0; i < num_of_hid_units; i++) { 
    net->b_in_to_hid[i] = (double) rand() / RAND_MAX;
    for(int j = 0; j < 784; j++) {
//    net->w_in_to_hid[i * 784 + j] = (double) rand() / RAND_MAX;     
      net->w_in_to_hid[i][j] = (double) rand() / RAND_MAX;     
    }
  }

  for(int i = 0; i < 10; i++){
    net->b_hid_to_out[i] = (double) rand() / RAND_MAX;
    for(int j = 0; j < num_of_hid_units; j++)
//    net->w_hid_to_out[i * num_of_hid_units + j] = (double) rand() / RAND_MAX;
      net->w_hid_to_out[i][j] = (double) rand() / RAND_MAX;
  }
}

void destructor_net(Network* net) {
  free(net->b_in_to_hid);
  free(net->b_hid_to_out);
  free(net->w_in_to_hid);
  free(net->w_hid_to_out);
  free(net);
}

void feed_forward(Network* net, double* input, uint cant_img, double* output) {
/*Return the output of the network if ``a`` is input.*/
  uint rows = net->num_of_hid_units;
  uint cols = 784;

  double* resProduct1 = (double*) malloc(rows * cant_img * sizeof(double));
  matrix_prod(net->w_in_to_hid, input, rows, cols, cant_img, resProduct1);

  double* z = (double*) malloc(rows * cant_img * sizeof(double));
  sum_vec(resProduct1, net->b_in_to_hid, rows, cant_img, z);

  double* hidden_state = (double*) malloc(rows * cant_img * sizeof(double));

  sigmoid_v(z, rows, cant_img, hidden_state);

  free(z);
  free(resProduct1);

  rows = 10;
  cols = net->num_of_hid_units;

  double* resProduct2 = (double*) malloc(rows * cant_img * sizeof(double));

  z = (double*) malloc(rows * cant_img * sizeof(double));

  matrix_prod(net->w_hid_to_out, hidden_state, rows, cols, cant_img, resProduct2);
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

void update_mini_batch(Network* net, Imagenes* minibatch, uint start, uint end) {
/*Update the network's weights and biases by applying
  gradient descent using backpropagation to a single mini batch.
  The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
  is the learning rate.*/
  uint h = net->num_of_hid_units;

  double* nabla_w_in_to_hid = (double*) malloc(h * 764 * sizeof(double));;      // h x 784
  double* nabla_b_in_to_hid = (double*) malloc(h * sizeof(double));   // h x 1
  double* nabla_w_hid_to_out = (double*) malloc(h * 10 * sizeof(double));;     // 10 x h
  double* nabla_b_hid_to_out = (double*) malloc(10 * sizeof(double)); // 10 x 1

  // dnb = delta nabla b
  // dnw = delta nabla w
  double* dnw_in_to_hid = (double*) malloc(h * 764 * sizeof(double));;      // h x 784
  double* dnb_in_to_hid = (double*) malloc(h * sizeof(double));   // h x 1
  double* dnw_hid_to_out = (double*) malloc(h * 10 * sizeof(double));;     // 10 x h
  double* dnb_hid_to_out = (double*) malloc(10 * sizeof(double)); // 10 x 1
  
  for(uint i = start; i < end; i++){
    backprop(net, minibatch->mat_tr, minibatch->res, start, end, dnb_hid_to_out, dnw_hid_to_out, dnb_in_to_hid, dnw_in_to_hid);
    sum_vec(nabla_w_in_to_hid, dnw_in_to_hid, h * 764, 1, nabla_w_in_to_hid);
    sum_vec(nabla_b_in_to_hid, dnb_in_to_hid, h, 1, nabla_b_in_to_hid);
    sum_vec(nabla_w_hid_to_out, dnw_hid_to_out, h * 10, 1, nabla_w_hid_to_out);
    sum_vec(nabla_b_hid_to_out, dnb_hid_to_out, 10, 1, nabla_b_hid_to_out);
  }

  uint mb_size = end-start;
  double eta = net->eta;
  update_weight(net->w_in_to_hid, nabla_w_in_to_hid, h * 764, mb_size, eta);
  update_weight(net->b_in_to_hid, nabla_b_in_to_hid, h, mb_size, eta);
  update_weight(net->w_hid_to_out, nabla_w_hid_to_out, 10 * h, mb_size, eta);
  update_weight(net->b_hid_to_out, nabla_b_hid_to_out, 10, mb_size, eta);  
}

void update_weight(double* w, double* nw, uint w_size, uint mb_size, double eta){
/*TO OPTIMIZE*/
  for(uint i = 0; i < w_size; i++){
    w[i] = w[i] - (eta/mb_size) * nw[i];
  }
}


void productoHadamard(double* matrix_1, double* matrix_2, uint rows, uint cols, double* output){
  // Las dos matrices de entrada y la de salida deben tener las mismas dimensiones
  for(uint i = 0; i < cols; i++){
    for(uint j = 0; j < rows; j++){
      output[i][j] += matrix_1[i][j] * matrix_2[i][j];
    }
  }
}


void backprop(Network* net, double* X, double* y, uint start, uint end, double* nb_hid_to_out , double* nw_hid_to_out, double* nb_in_to_hid, double* nw_in_to_hid){

/*Return a tuple ``(nabla_b, nabla_w)`` representing the
gradient for the cost function C_x.  ``nabla_b`` and
``nabla_w`` are layer-by-layer lists of numpy arrays, similar
to ``self.biases`` and ``self.weights``.*/

  int h = net->num_of_hid_units;

  double* activation0 = X;

  //--- FeedForward ---//

  // Ciclo 1

  uint cant_img = 1;
  uint inputUnits = 784;
  uint outputUnits = 10;

  double* resProduct1 = (double*) malloc(h * cant_img * sizeof(double));
  matrix_prod(net->w_in_to_hid, activation0, h, inputUnits, cant_img, resProduct1);

  double* z1 = (double*) malloc(h * cant_img * sizeof(double));
  sum_vec(resProduct1, net->bias_in_to_hid, h, cant_img, z1);

  double* activation1 = (double*) malloc(h * cant_img * sizeof(double));
  sigmoid_v(z1, h, cant_img, activation1);

  free(resProduct1);

  // Ciclo 2

  double* resProduct2 = (double*) malloc(outputUnits * cant_img * sizeof(double));
  matrix_prod(activation1, net->w_hid_to_out, outputUnits, h, cant_img, resProduct2);

  double* z2 = (double*) malloc(outputUnits * cant_img * sizeof(double));
  sum_vec(resProduct2, net->bias_hid_to_out, outputUnits, cant_img, z2);

  double* activation2 = (double*) malloc(outputUnits * cant_img * sizeof(double));
  sigmoid_v(z2, outputUnits, cant_img, activation2);  

  free(resProduct2);

  //--- BackProp ---//

  // Ciclo 1
  double* resProduct1 = (double*) malloc(outputUnits * cant_img * sizeof(double));

  cost_derivative(activation2, y, outputUnits, 1, double* resProduct1); 

  double* resProduct2 = (double*) malloc(outputUnits * cant_img * sizeof(double));

  sigmoid_prime_v(z2, outputUnits, 1, double* resProduct2);

  productoHadamard(resProduct1, resProduct2, outputUnits, 1, nb_hid_to_out);

  free(resProduct1);  
  free(resProduct2);

  matrix_prod(nb_hid_to_out, activations1, outputUnits, outputUnits, h, nw_hid_to_out);

  // Ciclo 2

  double* resProduct1 = (double*) malloc(h * cant_img * sizeof(double));

  sigmoid_prime_v(z1, h, 1, resProduct1);

  double* resProduct2 = (double*) malloc(h * cant_img * sizeof(double));

  matrix_prod(net->w_hid_to_out, nb_hid_to_out, uint rows, uint cols, 1, resProduct2);

  productoHadamard(resProduct1, resProduct2, h, 1, nb_in_to_hid);

  free(resProduct1);  
  free(resProduct2);

  matrix_prod(nb_in_to_hid, activation0, h, inputUnits, 1, nw_in_to_hid);

  //--- Libero memoria ---//

  free(z1);
  free(z2);

  free(activation1);
  free(activation2);

}

int evaluate(Network* net, double* test_data);
/*Return the number of test inputs for which the neural
  network outputs the correct result. Note that the neural
  network's output is assumed to be the index of whichever
  neuron in the final layer has the highest activation.*/

void cost_derivative(double* matrix, double* y, uint rows, uint cols, double* output) {
/*Return the vector of partial derivatives \partial C_x /
  \partial a for the output activations.*/
  for(uint k = 0; k < cols; k++){
    for(uint i = 0; i < rows; i++){
//      output[k * rows + i] = matrix[k * rows + i] - y[k * rows + i];
      output[k][i] = matrix[k][i] - y[k][i];
    }
  }
}

double sigmoid(double number){
/*The sigmoid function.*/
  return 1/(1 + exp(-number));
}

void sigmoid_v(double* matrix, uint rows, uint cols, double* output){
/*The sigmoid function.*/
  for(uint k = 0; k < cols; k++){
    for(uint i = 0; i < rows; i++) {
//      output[k * rows + i] = sigmoid(matrix[k * rows + i]);
      output[k][i] = sigmoid(matrix[k][i]);
    }
  }
}

double sigmoid_prime(double number){
/*The sigmoid function.*/
  return sigmoid(number)*(1-sigmoid(number));
}


void sigmoid_prime_v(double* matrix, uint rows, uint cols, double* output){
  for(uint k = 0; k < cols; k++){ 
    for(uint i = 0; i < rows; i++){
      double sigmoidea = sigmoid(matrix[k][i]);
      double minusOneSigmoid = 1 - sigmoidea;
      output[k][i] = minusOneSigmoid*sigmoidea;
    }
  }
}

void sum_vec(double* matrix, double* vector, uint vector_size, uint matrix_cols, double* output){
/* matrix_rows == vector_size */

  for(int k = 0; k < matrix_cols; k++){
    for(uint i = 0; i < vector_size; i++){
      output[k][i] = vector[k][i] + matrix[i];
    }
  }
}

void matrix_prod(double* matrix_1, double* matrix_2, uint matrix_1_rows, uint matrix_1_cols, uint matrix_2_cols, double* output){
/* matrix_1_cols == matrix_2_rows */

  for(uint k = 0; k < matrix_2_cols; k++) {
    for(uint i = 0; i < matrix_1_rows; i++){
      output[i][k] = 0;
      for(uint j = 0; j < matrix_1_cols; j++){
        output[i][k] += W[i][j] * X[j][k];
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
  initialize_net(net, 10, 0.3);

  if (training_data == NULL || test_data == NULL){
    printf("Error intentando leer data-sets de entrada\n");
    return 0;
  }

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