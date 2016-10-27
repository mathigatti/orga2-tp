#include <stdio.h>
#include <stdlib.h>

int* training_data; 
int* test_data;
int* training_labels;
int* test_labels;

struct Network {
  int num_layers;
  int* sizes;
  int* biases;
  int* weights;
};

double* feedForward(struct Network &net, double* input);
void SGD(struct Network &net);

int main (void){
  struct Network net;
  int sizes[3] = {17,72,73};
  net.sizes = sizes;

  for(int i = 0; i < 3; i++)
    printf("El tamaño %d es %d \n", i+1, net.sizes[i]);

  return 0;
}