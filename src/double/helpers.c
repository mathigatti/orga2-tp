#include "helpers.h"

Images* trainSetReader() {
  printf("Loading training set...\n");
  char buffer[150];
  char *record,*line;

  Images* Img = (Images*) malloc(sizeof(Images));
  Img->mat = (double*) malloc(IMGS_NUM * IMG_SIZE * sizeof(double));
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
  Img->mat = (double*) malloc(TEST_IMGS_NUM * IMG_SIZE * sizeof(double));
  Img->size = TEST_IMGS_NUM;

  FILE *fstream = fopen("../data/test_set.txt","r");
  if(fstream == NULL) {
    printf("\n file opening failed ");
    return NULL ;
  }

  for(int i = 0; i<TEST_IMGS_NUM; i++){
    //printf("Imagen %d\n",i);
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
  free(imgs);
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

double sigmoid(double number){
/*The sigmoid function.*/
  return 1/(1 + exp(-number));
}

double sigmoid_prime(double number){
  double sig = sigmoid(number); 
  return sig * (1 - sig);
}