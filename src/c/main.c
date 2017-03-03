#include "nn.h"

/*
The architechture of the network consist of
784 input units ---> a custom number of hidden units ---> 10 output units
in a full-connected style
*/

void initialize_net(Network* net, uint num_of_hid_units, double eta){
  srand(time(NULL));
  /*
  for(uint i = 0; i < 1000; i++){
    printf("%f\n", (double) rand() / RAND_MAX);
  }
  */
  net->eta = eta;
  net->num_of_hid_units = num_of_hid_units;
  net->bias_in_to_hid = (double*) malloc(num_of_hid_units*sizeof(double));
  net->bias_hid_to_out = (double*) malloc(10*sizeof(double));
  net->w_in_to_hid = (double*) malloc(784 * num_of_hid_units * sizeof(double));
  net->w_hid_to_out = (double*) malloc(10 * num_of_hid_units * sizeof(double));
  
  for(int i = 0; i < num_of_hid_units; i++) { 
    net->bias_in_to_hid[i] = (double) rand() / RAND_MAX;
    net->bias_in_to_hid[i] /= 10.0;
    for(int j = 0; j < 784; j++) {
      net->w_in_to_hid[i * 784 + j] = (double) rand() / RAND_MAX;     
      net->w_in_to_hid[i * 784 + j] /= 10.0;     
    }
  }

  for(int i = 0; i < 10; i++){
    net->bias_hid_to_out[i] = (double) rand() / RAND_MAX;
    net->bias_hid_to_out[i] /= 10.0;
    for(int j = 0; j < num_of_hid_units; j++) {
      net->w_hid_to_out[i * num_of_hid_units + j] = (double) rand() / RAND_MAX;
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

void feed_forward(Network* net, double* input, uint cant_img, double* output) {
/*Return the output of the network if ``a`` is input.*/
  double* input_tr = (double*) malloc(784 * cant_img * sizeof(double));
  transpose(input, cant_img, 784, input_tr);

  uint rows = net->num_of_hid_units;
  uint cols = 784;
  cant_img = 1;

  double* resProduct1 = (double*) malloc(rows * cant_img * sizeof(double));
  matrix_prod(net->w_in_to_hid, input_tr, rows, cols, cant_img, resProduct1);

  double* z = (double*) malloc(rows * cant_img * sizeof(double));
  mat_plus_vec(resProduct1, net->bias_in_to_hid, rows, cant_img, z);

  double* hidden_state = (double*) malloc(rows * cant_img * sizeof(double));

  sigmoid_v(z, rows, cant_img, hidden_state);

  free(input_tr);
  free(z);
  free(resProduct1);

  rows = 10;
  cols = net->num_of_hid_units;

  double* resProduct2 = (double*) malloc(rows * cant_img * sizeof(double));

  z = (double*) malloc(rows * cant_img * sizeof(double));

  matrix_prod(net->w_hid_to_out, hidden_state, rows, cols, cant_img, resProduct2);
  mat_plus_vec(resProduct2, net->bias_hid_to_out, rows, cant_img, z);

  sigmoid_v(z, rows, cant_img, output);
 
  free(resProduct2);
  free(hidden_state);
  free(z);
}

void SGD(Network* net, Images* training_data, uint epochs, uint mini_batch_size, double eta){
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
    //printf("Shuffle exitoso\n");
    for(uint j = 0; j < n; j += mini_batch_size){
      update_mini_batch(net, training_data, j, j + mini_batch_size);
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

  double* nabla_w_in_to_hid = (double*) calloc(h * 784, sizeof(double));      // h x 784
  double* nabla_b_in_to_hid = (double*) calloc(h, sizeof(double));   // h x 1
  double* nabla_w_hid_to_out = (double*) calloc(h * 10, sizeof(double));     // 10 x h
  double* nabla_b_hid_to_out = (double*) calloc(10, sizeof(double)); // 10 x 1

  // dnb = delta nabla b
  // dnw = delta nabla w
  double* dnw_in_to_hid = (double*) malloc(h * 784 * sizeof(double));    // h x 784
  double* dnb_in_to_hid = (double*) malloc(h * sizeof(double));   // h x 1
  double* dnw_hid_to_out = (double*) malloc(h * 10 * sizeof(double));     // 10 x h
  double* dnb_hid_to_out = (double*) malloc(10 * sizeof(double)); // 10 x 1
  
  for(uint i = start; i < end; i++){
    backprop(net, &minibatch->mat[i*784], minibatch->res[i], dnw_in_to_hid, dnb_in_to_hid, dnw_hid_to_out, dnb_hid_to_out);
    mat_plus_vec(nabla_w_in_to_hid, dnw_in_to_hid, h * 784, 1, nabla_w_in_to_hid);
    mat_plus_vec(nabla_b_in_to_hid, dnb_in_to_hid, h, 1, nabla_b_in_to_hid);
    mat_plus_vec(nabla_w_hid_to_out, dnw_hid_to_out, h * 10, 1, nabla_w_hid_to_out);
    mat_plus_vec(nabla_b_hid_to_out, dnb_hid_to_out, 10, 1, nabla_b_hid_to_out);
  }

  // Free memory for delta nablas
  free(dnw_in_to_hid);
  free(dnb_in_to_hid);
  free(dnw_hid_to_out);
  free(dnb_hid_to_out);

  uint mb_size = end-start;
  double eta = net->eta;
  update_weight(net->w_in_to_hid, nabla_w_in_to_hid, h * 784, mb_size, eta);
  update_weight(net->bias_in_to_hid, nabla_b_in_to_hid, h, mb_size, eta);
  update_weight(net->w_hid_to_out, nabla_w_hid_to_out, 10 * h, mb_size, eta);
  update_weight(net->bias_hid_to_out, nabla_b_hid_to_out, 10, mb_size, eta);

  // Free memory for nablas
  free(nabla_w_in_to_hid);
  free(nabla_b_in_to_hid);
  free(nabla_w_hid_to_out);
  free(nabla_b_hid_to_out);  
}

void update_weight(double* w, double* nw, uint w_size, uint mb_size, double eta){
/*TO OPTIMIZE*/
  for(uint i = 0; i < w_size; i++){
    w[i] -= (eta/mb_size) * nw[i];
  }
}


void backprop(Network* net, double* input, int target, double* nw_in_to_hid, double* nb_in_to_hid, double* nw_hid_to_out, double* nb_hid_to_out){
/*Return a tuple ``(nabla_b, nabla_w)`` representing the
gradient for the cost function C_x.  ``nabla_b`` and
``nabla_w`` are layer-by-layer lists of numpy arrays, similar
to ``self.biases`` and ``self.weights``.*/

  int h = net->num_of_hid_units;

  double* activation0 = input;

  //--- FeedForward ---//

  // Ciclo 1

  uint cant_img = 1;
  uint inputUnits = 784;
  uint outputUnits = 10;

  double* resProduct1 = (double*) malloc(h * cant_img * sizeof(double));
  matrix_prod(net->w_in_to_hid, activation0, h, inputUnits, cant_img, resProduct1);

  double* z1 = (double*) malloc(h * cant_img * sizeof(double));
  mat_plus_vec(resProduct1, net->bias_in_to_hid, h, cant_img, z1);

  double* activation1 = (double*) malloc(h * cant_img * sizeof(double));
  sigmoid_v(z1, h, cant_img, activation1);

  free(resProduct1);

  // Ciclo 2

  double* resProduct2 = (double*) malloc(outputUnits * cant_img * sizeof(double));
  matrix_prod(net->w_hid_to_out, activation1, outputUnits, h, cant_img, resProduct2);

  double* z2 = (double*) malloc(outputUnits * cant_img * sizeof(double));
  mat_plus_vec(resProduct2, net->bias_hid_to_out, outputUnits, cant_img, z2);

  double* activation2 = (double*) malloc(outputUnits * cant_img * sizeof(double));
  sigmoid_v(z2, outputUnits, cant_img, activation2);  

  free(resProduct2);

  //--- BackProp ---//

  // Ciclo 1

  resProduct1 = (double*) malloc(outputUnits * cant_img * sizeof(double));

  double y[10] = {[0 ... 9] = 0};
  y[target] = 1;

  // (y - t)
  cost_derivative(activation2, y, outputUnits, 1, resProduct1); 

  resProduct2 = (double*) malloc(outputUnits * cant_img * sizeof(double));

  // y(1-y)
  sigmoid_prime_v(z2, outputUnits, 1, resProduct2);

  // y(1-y)(y-t)
  hadamardProduct(resProduct1, resProduct2, outputUnits, 1, nb_hid_to_out);

  free(resProduct1);  
  free(resProduct2);

  // xy(1-y)(y-t)
  matrix_prod(nb_hid_to_out, activation1, outputUnits, 1, h, nw_hid_to_out);

  // Hasta aca me cierra bien
  // Ciclo 2

  resProduct1 = (double*) malloc(h * cant_img * sizeof(double));

  sigmoid_prime_v(z1, h, 1, resProduct1);

  resProduct2 = (double*) malloc(h * cant_img * sizeof(double));

  // Hay que transponer net->w_hid_to_out 
  double* aux = (double*) malloc(h * 10 * sizeof(double));
  transpose(net->w_hid_to_out, 10, h, aux);
  matrix_prod(aux, nb_hid_to_out, h, 10, 1, resProduct2);
  free(aux);

  hadamardProduct(resProduct1, resProduct2, h, 1, nb_in_to_hid);

  free(resProduct1);  
  free(resProduct2);

  matrix_prod(nb_in_to_hid, activation0, h, 1, inputUnits, nw_in_to_hid);

  //--- Libero memoria ---//

  free(z1);
  free(z2);

  free(activation1);
  free(activation2);

}


double evaluate(Network* net, Images* test_data){
/*Return the accuracy over test_data*/
  int hits = 0;
  double* res = (double*) malloc(10 * sizeof(double));

  for(uint i = 0; i < test_data->size; i++) {
    feed_forward(net, &test_data->mat[i * 784], 1, res);
    int y = max_arg(res, 10);
    if(y == test_data->res[i]) {
      hits++;
    }
  }
  free(res);

  return hits / (double) test_data->size;
}

int max_arg(double* vector, uint n) {
  int maxIndex = 0;
  double maxValue = vector[maxIndex]; 
  for(int i = 1; i < n; i++){
    if(maxValue < vector[i]) {
      maxIndex = i;
      maxValue = vector[i];
    }
  }
  return maxIndex;
}

void transpose(double* matrix, uint n, uint m, double* output){
  /* NOTA: n y m no tienen que coincidir forzosamente con la cantidad de 
           filas y columnas real de matrix. Por ejemplo, si matrix es pxm
           con n < p, output sera la matriz que tenga por columnas las 
           primeras n filas de matrix. Esto es util a la hora de usar mini 
           batches.
  */
  for(uint i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      output[j * n + i] = matrix[i * m + j];
    }
  }
}

void hadamardProduct(double* matrix1, double* matrix2, uint n, uint m, double* output){
/* matrix1 and matrix2 are nxm*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = matrix1[i * m + j] * matrix2[i * m + j];
    }
  }
}

/*
void cost_derivative(double* matrix, double* matrix2, uint n, uint m, double* output) {
//Return the vector of partial derivatives \partial C_x /
//partial a for the output activations.
// Normalmente m = 1
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = matrix[i * m + j] - matrix2[i * m + j];
    }
  }
}
*/

double sigmoid(double number){
/*The sigmoid function.*/
  return 1/(1 + exp(-number));
}

void sigmoid_v(double* matrix, uint n, uint m, double* output){
/*The sigmoid function.*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++) {
      output[i * m + j] = sigmoid(matrix[i * m + j]);
    }
  }
}

double sigmoid_prime(double number){
  double sig = sigmoid(number); 
  return sig * (1 - sig);
}


void sigmoid_prime_v(double* matrix, uint n, uint m, double* output){
  double sig;
  double minusOneSig;
  for(uint i = 0; i < n; i++){ 
    for(uint j = 0; j < m; j++){
      sig = sigmoid(matrix[i * m + j]);
      minusOneSig = 1 - sig;
      output[i * m + j] = minusOneSig * sig;
    }
  }
}



void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output){
// |vector| == n

  for(int i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = vector[i] + matrix[i * m + j];
    }
  }
}

void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output){
/* matrix1 is nxm */
/* matrix2 is mxl */
/* output is nxl */
  for(uint i = 0; i < n; i++) {
    for(uint j = 0; j < l; j++){
      output[i * l + j] = 0;
      for(uint k = 0; k < m; k++){
        output[i * l + j] += matrix1[i * m + k] * matrix2[k * l + j];
      }
    }
  }
}


void random_shuffle(Images* batch) {
  size_t n = IMGS_NUM;
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);

      // printImg(&batch->mat[i * 784]);
      // printImg(&batch->mat[j * 784]);
      
      // We permute the rows of batch.mat
      double temp_pixel;
      for(uint k = 0; k < 784; k++) {
        temp_pixel = batch->mat[j * 784 + k];
        batch->mat[j * 784 + k] = batch->mat[i * 784 + k];
        batch->mat[i * 784 + k] = temp_pixel;
      }

      // Finally, let's permute the targets
      int temp_res = batch->res[j];
      batch->res[j] = batch->res[i];
      batch->res[i] = temp_res;
      // printImg(&batch->mat[i * 784]);
      // printImg(&batch->mat[j * 784]);
      // printf("*****************************\n");
    }
  }
}

void printImg(double* img) {
  for(uint i = 0; i < 28; i++) {
    for(uint j = 0; j < 28; j++) {
      if(img[i * 28 + j] >= 0.45) {
        printf("X");
      } else {
        printf(" ");
      }
    }
    printf("\n");
  }
}

void printMatrix(double* matrix, int n, int m) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      printf("%.3f ", matrix[i * m + j]);
    }
    printf("\n");
  }
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
  double* res = (double*) malloc(10 * sizeof(double));
  
  SGD(net, training_data, 28, MINI_BATCH_SIZE, net->eta);

  feed_forward(net, &test_data->mat[3 * 784], 1, res);
  // printImg(&test_data->mat[3 * 784]);
  // printf("Target: %d\n", test_data->res[3]);
  for(int i = 0; i < 10; i++){
    printf("Valor para %d: %f\n", i, res[i]);
  }
  printf("Target: %d\n", test_data->res[3]);

  //testeo feedforward con un 1 artificial
  double input[784] = {[0 ... 783] = 0};
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

  return 0;
}

/*
  ALTAS CHANCES de que todos los valores que hay en las matrices de pesos 
  sean 0 despues del entrenamiento. Eso explicaria porque siempre se 
  obtiene el mismo resultado independientemente del input.
  Siempre se estaria obteniendo el bias de hidden to output layer.

  UPDATE: Los pesos son distintos a 0. Logre avanzar un poco, y llegue a la
  conclusión de que los pesos son tan grandes en la capa de entrada que provocan que el hidden state siempre sea 1.0 para todas las unidades. Eso 
  explicaria el por que no se depende del input. Lo que habria que ver entonces es porque no aprende pesos acordes a esto.

  Finalmente resulto ser que el codigo estaba bien. El problema en realidad 
  era que los valores con los que se inicializaban las matrices de pesos eran muy grandes (pese a ser random). Sospecho que la razon de que este problema no surgiera con la version de python es la diferencia en la calidad del sampleo. Mi solucion "artesanal" fue reescalar las valores sampleados, dividiendolos por 10.0
  Este cambio permitio replicar el accuracy expuesto por el autor del algoritmo en python

*/