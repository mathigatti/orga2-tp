#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define tamanioImagen 784
#define tamanioRes 1
#define cantImagenes 100
#define tamanioTotalRecorrer (tamanioRes + tamanioImagen + 1)

struct Imagenes {
	float mat[cantImagenes*tamanioImagen];
	int res[cantImagenes];
	int cantImg;
};


int main()
{
   char buffer[150];
   char *record,*line;

   struct Imagenes Img;
   Img.cantImg = cantImagenes;

   FILE *fstream = fopen("/home/mathi/Desktop/train_set.txt","r");
   if(fstream == NULL)
   {
      printf("\n file opening failed ");
      return -1 ;
   }

	for(int i = 0; i<cantImagenes; i++){
		printf("Imagen %d\n",i);
		for(int j = 0; j<tamanioImagen; j++){
			line=fgets(buffer,sizeof(buffer),fstream);
			record = strtok(line,"\n");
			printf("%s\n",record);
			Img.mat[i*tamanioImagen+j] = atof(record);
			record = strtok(NULL,"\n");				
	   	}
		line=fgets(buffer,sizeof(buffer),fstream);
		record = strtok(line,"\n");
		printf("%s\n",record);
		Img.res[i] = atoi(record);
		record = strtok(NULL,"\n");
    }
}