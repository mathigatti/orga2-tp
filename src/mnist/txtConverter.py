import cPickle, gzip, numpy

def formatearEspaciado(imagen,numero,myfile):
	for pixel in imagen:
	    myfile.write(str(pixel)+'\n')
	myfile.write(str(numero)+'\n')

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
# len de train_set:  50000
# len de valid_set:  10000
# len de test_set:  10000
train_set, valid_set, test_set = cPickle.load(f)
# with open("train_set.txt", "a") as myfile:
# 	cantImagenes = 50000
# 	for i in range(cantImagenes):
# 		imagen = train_set[0][i]
# 		res = train_set[1][i]
# 		formatearEspaciado(imagen,res,myfile)

with open("test_set.txt", "a") as myfile:
  cantImagenes = 10000
  for i in range(cantImagenes):
    imagen = test_set[0][i]
    res = test_set[1][i]
    formatearEspaciado(imagen,res,myfile)
f.close()