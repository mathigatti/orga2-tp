#include "helpers.h"

Images* trainSetReader() {
  printf("Loading training set...\n");
  char buffer[150];
  char *record,*line;

  Images* Img = (Images*) malloc(sizeof(Images));
  Img->mat = (double*) malloc(IMGS_NUM * IMG_SIZE * sizeof(double));
  Img->res = (int*) malloc(IMGS_NUM * sizeof(double));
  Img->size = IMGS_NUM;

  FILE *fstream = fopen("data/train_set.txt","r");
  if(fstream == NULL) {
    printf("\n file opening failed ");
    return NULL ;
  }

	for(int i = 0; i<IMGS_NUM; i++){
		for(int j = 0; j<IMG_SIZE; j++){
			line=fgets(buffer,sizeof(buffer),fstream);
			record = strtok(line,"\n");

      Img->mat[i * IMG_SIZE + j] = atof(record);
			record = strtok(NULL,"\n");				
	  }
		line=fgets(buffer,sizeof(buffer),fstream);
		record = strtok(line,"\n");

		Img->res[i] = atoi(record);
		record = strtok(NULL,"\n");
  }
  fclose(fstream);
  return Img;
}

Images* testSetReader() {
  printf("Loading test set...\n");
  char buffer[150];
  char *record,*line;

  Images* Img = (Images*) malloc(sizeof(Images));
  Img->mat = (double*) malloc(TEST_IMGS_NUM * IMG_SIZE * sizeof(double));
  Img->res = (int*) malloc(TEST_IMGS_NUM * sizeof(double));
  Img->size = TEST_IMGS_NUM;

  FILE *fstream = fopen("data/test_set.txt","r");
  if(fstream == NULL) {
    printf("\n file opening failed ");
    return NULL ;
  }

  for(int i = 0; i<TEST_IMGS_NUM; i++){

    for(int j = 0; j<IMG_SIZE; j++){
      line=fgets(buffer,sizeof(buffer),fstream);
      record = strtok(line,"\n");

      Img->mat[i * IMG_SIZE + j] = atof(record);
      record = strtok(NULL,"\n");       
      }
    line=fgets(buffer,sizeof(buffer),fstream);
    record = strtok(line,"\n");

    Img->res[i] = atoi(record);
    record = strtok(NULL,"\n");
  }

  fclose(fstream);
  return Img;
}

double* loadTestImage(double* matrix, const char* inputImage) {
  printf("Loading test image...\n");
  char buffer[150];
  char *record,*line;

  FILE *fstream = fopen(inputImage,"r");
  if(fstream == NULL) {
    printf("\n file opening failed\n");
    return NULL ;
  }

  for(int j = 0; j<IMG_SIZE; j++){
    line=fgets(buffer,sizeof(buffer),fstream);
    record = strtok(line,"\n");
    matrix[j] = atof(record);
  }

  fclose(fstream);
}

void imagesDestructor(Images* imgs) {
  free(imgs->mat);
  free(imgs->res);
  free(imgs);
}

void random_shuffle(Images* batch) {
  size_t n = batch->size;
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);

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
    }
  }
}

void printImg(double* img) {
  for(uint i = 0; i < 28; i++) {
    for(uint j = 0; j < 28; j++) {
      if(img[i * 28 + j] >= 0.45) {
        printf("O");
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

double sigmoid(double number){
/*The sigmoid function.*/
  return 1/(1 + exp(-number));
}

double sigmoid_prime(double number){
  double sig = sigmoid(number); 
  return sig * (1 - sig);
}

// A Implementar en ASM

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

void sigmoid_v(double* matrix, uint n, uint m, double* output){
/*The sigmoid function.*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++) {
      output[i * m + j] = sigmoid(matrix[i * m + j]);
    }
  }
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

void randomVector(uint size, double* vector, uint randMax){
  for (uint i = 0; i < size; i++){
      vector[i] = (double) rand() / RAND_MAX;
  }
}

void randomMatrix(double* matrix, uint n, uint m){
  for (uint i = 0; i < n; i++){
    for (uint j = 0; j < m; j++){
      matrix[i * m + j] = (double) rand() / RAND_MAX;
    }
  }
}

void compress(double* matrix, uint n, uint m, double* output) {
  // output.length() = n
  for (int i = 0; i < n; i++) {
    output[i] = 0.0;
    for (int j = 0; j < m; j++) {
      output[i] += matrix[i * m + j];
    }
  }
}

void mat_plus_vec(double* matrix, double* vector, uint n, uint m, double* output){
// |vector| == n
  for(int i = 0; i < n; i++){
    double val = vector[i];
    for (int j = 0; j < m; j++) {
      output[i * m + j] = val + matrix[i * m + j];
    }
  }
}