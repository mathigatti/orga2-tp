#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define IMG_SIZE 784
#define RES_SIZE 1
#define IMGS_NUM 10000
#define tamanioTotalRecorrer (RES_SIZE + IMG_SIZE + 1)

typedef struct Imagenes {
  double mat[IMGS_NUM * IMG_SIZE];
	double mat_tr[IMG_SIZE * IMGS_NUM];
	int res[IMGS_NUM];
	int cantImg;
} Imagenes;


Imagenes* trainSetReader() {
  char buffer[150];
  char *record,*line;

  Imagenes* Img = (Imagenes*) malloc(sizeof(Imagenes));
  Img->cantImg = IMGS_NUM;

  FILE *fstream = fopen("../data/train_set.txt","r");
  if(fstream == NULL) {
    printf("\n file opening failed ");
    return NULL ;
  }

	for(int i = 0; i<IMGS_NUM; i++){
		//printf("Imagen %d\n",i);
		for(int j = 0; j<IMG_SIZE; j++){
			line=fgets(buffer,sizeof(buffer),fstream);
			record = strtok(line,"\n");
			//printf("%s\n",record);
      Img->mat[i * IMG_SIZE + j] = atof(record);
			Img->mat_tr[j * IMGS_NUM + i] = atof(record);
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

Imagenes* testSetReader() {
  char buffer[150];
  char *record,*line;

  Imagenes* Img = (Imagenes*) malloc(sizeof(Imagenes));
  Img->cantImg = IMGS_NUM;

  FILE *fstream = fopen("../data/test_set.txt","r");
  if(fstream == NULL) {
    printf("\n file opening failed ");
    return NULL ;
  }

  for(int i = 0; i<IMGS_NUM; i++){
    //printf("Imagen %d\n",i);
    for(int j = 0; j<IMG_SIZE; j++){
      line=fgets(buffer,sizeof(buffer),fstream);
      record = strtok(line,"\n");
      //printf("%s\n",record);
      Img->mat[i * IMG_SIZE + j] = atof(record);
      Img->mat_tr[j * IMGS_NUM + i] = atof(record);
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
