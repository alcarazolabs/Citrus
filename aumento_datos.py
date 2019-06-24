from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from imutils import paths
import argparse
import cv2
import os
import numpy as np

#obtener argumentos desde consola
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Ruta del dataset de entrada")
ap.add_argument("-o", "--output", required=True, help="Ruta para guardar el dataset aunmentado")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))
salida = args["output"]
"""
datagen = ImageDataGenerator(
        rotation_range=40, #0-180°
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0,
        horizontal_flip=True,
        fill_mode='nearest')
"""
datagen = ImageDataGenerator(
        rotation_range=40, #0-180°
        horizontal_flip=True,
        fill_mode='nearest')

j = 0
numero = 1
for (i, imagePath) in enumerate(imagePaths):
	#cargar la imagen desde el disco
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	x = img_to_array(image)
	x = x.reshape((1,) + x.shape)
	# /path/to/dataset/{class}/{image}.jpg
	label = imagePath.split(os.path.sep)[0]
	clase = label.split("/")
	clase = clase[2]

	nombre = ""
	if clase == 'maduro':
		nombre = "maduro"+"_"+str(numero)
	elif clase == 'pinton':
		nombre = "pinton"+"_"+str(numero)
	elif clase == 'verde':
		nombre = "verde"+"_"+str(numero)
	if numero == 350:
		numero = 0
	print(nombre)
	numero +=1
	for batch in datagen.flow(x, batch_size=1,
								save_to_dir=salida, save_prefix=nombre, save_format='jpg'):
		j += 1
		if j == 20: #20 imagenesp por imagen.
			j = 0
			break  #De otra manera el generador seria infinito.
              
print("[INFO] data augmentation realizado exitosamente")