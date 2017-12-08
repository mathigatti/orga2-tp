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
  float* mat;
  int* res;
  int size; //Number of images
} Images;

Images* trainSetReader();

Images* testSetReader();

float* loadTestImage(float* matrix, const char* inputImage);

void imagesDestructor(Images* imgs);

float sigmoid(float z);

float sigmoid_prime(float z);

void printImg(float* img);

void printMatrix(float* matrix, int n, int m);

void random_shuffle(Images* batch);

int max_arg(float* vector, uint n);

void sigmoid_v(float* matrix, uint n, uint m, float* output);

void sigmoid_prime_v(float* matrix, uint n, uint m, float* output);

void transpose(float* matrix, uint n, uint m, float* output);

void randomVector(uint size, float* vector, uint randMax);

void randomMatrix(float* matrix, uint n, uint m);

void compress(float* matrix, uint n, uint m, float* output);

void mat_plus_vec(float* matrix, float* vector, uint n, uint m, float* output);

#endif