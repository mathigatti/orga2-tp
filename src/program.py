
import sys
import matplotlib.image as img
import subprocess
import os

programs = {"c": "c_version", "asm": "asm_version"}

def formatearEspaciado(matrix,myfile):
	for row in matrix:
		for value in row:
			greySclale = 1.0 - value.mean()
			myfile.write(str(greySclale)+'\n')

def main(argv):
	if len(argv) < 3:
		msg =  "Input Error\nEjemplo de ejecucion correcto:\n\tpython program.py asm float test_image.png"
		raise Exception(msg)
	languageType = argv[0]
	numberType = argv[1]
	imagePath = argv[2]
	image = img.imread(imagePath)

	filename, file_extension = os.path.splitext(imagePath)
	txtImage = filename + ".txt"

	with open(txtImage, "w") as myfile:
		formatearEspaciado(image, myfile)

	programPath = "./" + numberType + "/"
	subprocess.call(["make"], cwd=programPath)
	subprocess.call([programPath + programs[languageType], txtImage])
	subprocess.call(["rm","txtImage"], cwd=programPath)

if __name__ == "__main__":
	main(sys.argv[1:])