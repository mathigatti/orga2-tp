#ifndef helpers_h
#define helpers_h

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#define IMG_SIZE 784
#define RES_SIZE 1
#define IMGS_NUM 50000
#define TEST_IMGS_NUM 10000
#define tamanioTotalRecorrer (RES_SIZE + IMG_SIZE + 1)

typedef unsigned int uint;

typedef struct Images {
  double* mat;
  int res[IMGS_NUM];
  int size; //Number of images
} Images;

Images* trainSetReader();

Images* testSetReader();

void imagesDestructor(Images* imgs);

double sigmoid(double z);

double sigmoid_prime(double z);

void printImg(double* img);

void printMatrix(double* matrix, int n, int m);

void random_shuffle(Images* batch);

// A implementar en asm

//MATHI
void sigmoid_v(double* matrix, uint n, uint m, double* output);

void sigmoid_prime_v(double* matrix, uint n, uint m, double* output);

void transpose(double* matrix, uint n, uint m, double* output);

void hadamardProduct(double* matrix1, double* matrix2, uint n, uint m, double* output);

//MANU
void matrix_prod(double* matrix1, double* matrix2, uint n, uint m, uint l, double* output);

int max_arg(double* vector, uint n);

// Posiblemente se podria cambiar para que tome un vector directamente
// i.e. hacer suma de vector-vector
void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output);

void update_weight(double* w, double* nw, uint w_size, uint mb_size, double eta);

#endif