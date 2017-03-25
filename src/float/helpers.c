#include "helpers.h"

Images* trainSetReader() {
  printf("Loading training set...\n");
  char buffer[150];
  char *record,*line;

  Images* Img = (Images*) malloc(sizeof(Images));
  Img->mat = (float*) malloc(IMGS_NUM * IMG_SIZE * sizeof(float));
  Img->res = (int*) malloc(IMGS_NUM * sizeof(float));
  Img->size = IMGS_NUM;

  FILE *fstream = fopen("../data/train_set.txt","r");
  if(fstream == NULL) {
    printf("\n file opening failed ");
    return NULL ;
  }

	for(int i = 0; i<IMGS_NUM; i++){
		for(int j = 0; j<IMG_SIZE; j++){
			line=fgets(buffer,sizeof(buffer),fstream);
			record = strtok(line,"\n");
			//printf("%s\n",record);
      Img->mat[i * IMG_SIZE + j] = atof(record);
			record = strtok(NULL,"\n");				
	   	}
		line=fgets(buffer,sizeof(buffer),fstream);
		record = strtok(line,"\n");
		//printf("%s\n",record);
		Img->res[i] = atoi(record);
		record = strtok(NULL,"\n");
    //printf("Imagen %d\n",i);
  }
  fclose(fstream);
  return Img;
}

Images* testSetReader() {
  printf("Loading test set...\n");
  char buffer[150];
  char *record,*line;

  Images* Img = (Images*) malloc(sizeof(Images));
  Img->mat = (float*) malloc(TEST_IMGS_NUM * IMG_SIZE * sizeof(float));
  Img->res = (int*) malloc(TEST_IMGS_NUM * sizeof(float));
  Img->size = TEST_IMGS_NUM;

  FILE *fstream = fopen("../data/test_set.txt","r");
  if(fstream == NULL) {
    printf("\n file opening failed ");
    return NULL ;
  }

  for(int i = 0; i<TEST_IMGS_NUM; i++){
    // printf("Imagen %d\n",i);
    for(int j = 0; j<IMG_SIZE; j++){
      line=fgets(buffer,sizeof(buffer),fstream);
      record = strtok(line,"\n");
      //printf("%s\n",record);
      Img->mat[i * IMG_SIZE + j] = atof(record);
      record = strtok(NULL,"\n");       
      }
    line=fgets(buffer,sizeof(buffer),fstream);
    record = strtok(line,"\n");
    //printf("%s\n",record);
    Img->res[i] = atoi(record);
    record = strtok(NULL,"\n");
  }

  fclose(fstream);
  return Img;
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

      // printImg(&batch->mat[i * 784]);
      // printImg(&batch->mat[j * 784]);
      
      // We permute the rows of batch.mat
      float temp_pixel;
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

void printImg(float* img) {
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

void printMatrix(float* matrix, int n, int m) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      printf("%.3f ", matrix[i * m + j]);
    }
    printf("\n");
  }
}

float sigmoid(float number){
/*The sigmoid function.*/
  return 1/(1 + exp(-number));
}

float sigmoid_prime(float number){
  float sig = sigmoid(number); 
  return sig * (1 - sig);
}


// A Implementar en ASM

int max_arg(float* vector, uint n) {
  int maxIndex = 0;
  float maxValue = vector[maxIndex]; 
  for(int i = 1; i < n; i++){
    if(maxValue < vector[i]) {
      maxIndex = i;
      maxValue = vector[i];
    }
  }
  return maxIndex;
}

void transpose(float* matrix, uint n, uint m, float* output){
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

void hadamardProduct(float* matrix1, float* matrix2, uint n, uint m, float* output){
/* matrix1 and matrix2 are nxm*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++){
      output[i * m + j] = matrix1[i * m + j] * matrix2[i * m + j];
    }
  }
}

void sigmoid_v(float* matrix, uint n, uint m, float* output){
/*The sigmoid function.*/
  for(uint i = 0; i < n; i++){
    for(uint j = 0; j < m; j++) {
      output[i * m + j] = sigmoid(matrix[i * m + j]);
    }
  }
}

void sigmoid_prime_v(float* matrix, uint n, uint m, float* output){
  float sig;
  float minusOneSig;
  for(uint i = 0; i < n; i++){ 
    for(uint j = 0; j < m; j++){
      sig = sigmoid(matrix[i * m + j]);
      minusOneSig = 1 - sig;
      output[i * m + j] = minusOneSig * sig;
    }
  }
}




