import cPickle, gzip, numpy

def formatearEspaciado(imagen,numero,myfile):
	for pixel in imagen:
	    myfile.write(str(pixel)+'\n')
	myfile.write(str(numero)+'\n')

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
with open("train_set.txt", "a") as myfile:
	cantImagenes = 50000
	for i in range(cantImagenes):
		imagen = train_set[0][i]
		res = train_set[1][i]
		formatearEspaciado(imagen,res,myfile)
f.close()