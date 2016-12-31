#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define tamanioImagen 784
#define tamanioRes 1
#define cantImagenes 100
#define tamanioTotalRecorrer (tamanioRes + tamanioImagen + 1)

typedef struct Imagenes {
  double mat[cantImagenes*tamanioImagen];
	double mat_tr[cantImagenes*tamanioImagen];
	int res[cantImagenes];
	int cantImg;
} Imagenes;


Imagenes* trainSetReader() {
   char buffer[150];
   char *record,*line;

   Imagenes* Img = (Imagenes*) malloc(sizeof(Imagenes));
   Img->cantImg = cantImagenes;

   FILE *fstream = fopen("../data/train_set.txt","r");
   if(fstream == NULL) {
      printf("\n file opening failed ");
      return NULL ;
   }

	for(int i = 0; i<cantImagenes; i++){
		//printf("Imagen %d\n",i);
		for(int j = 0; j<tamanioImagen; j++){
			line=fgets(buffer,sizeof(buffer),fstream);
			record = strtok(line,"\n");
			//printf("%s\n",record);
      Img->mat[i*tamanioImagen+j] = atof(record);
			Img->mat_tr[j*cantImagenes+i] = atof(record);
			record = strtok(NULL,"\n");				
	   	}
		line=fgets(buffer,sizeof(buffer),fstream);
		record = strtok(line,"\n");
		//printf("%s\n",record);
		Img->res[i] = atoi(record);
		record = strtok(NULL,"\n");
    }

  return Img;
}

Imagenes* testSetReader() {
   char buffer[150];
   char *record,*line;

   Imagenes* Img = (Imagenes*) malloc(sizeof(Imagenes));
   Img->cantImg = cantImagenes;

   FILE *fstream = fopen("../data/test_set.txt","r");
   if(fstream == NULL) {
      printf("\n file opening failed ");
      return NULL ;
   }

  for(int i = 0; i<cantImagenes; i++){
    //printf("Imagen %d\n",i);
    for(int j = 0; j<tamanioImagen; j++){
      line=fgets(buffer,sizeof(buffer),fstream);
      record = strtok(line,"\n");
      //printf("%s\n",record);
      Img->mat[i*tamanioImagen+j] = atof(record);
      Img->mat_tr[j*cantImagenes+i] = atof(record);
      record = strtok(NULL,"\n");       
      }
    line=fgets(buffer,sizeof(buffer),fstream);
    record = strtok(line,"\n");
    //printf("%s\n",record);
    Img->res[i] = atoi(record);
    record = strtok(NULL,"\n");
    }

  return Img;
}
