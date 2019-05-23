#-------------------------------------------------------
# EXTRACCIÓN DE CARACTERISTICAS DE MOMENTOS DE COLOR
#--------------------------------------------------------
# Importar Paquetes Necesarios
from sklearn.preprocessing import LabelEncoder #Codificador de clases
import numpy as np
import cv2
import os
import h5py #Libreria para guardar los vectores de características en formato H5/HDF5.
from imutils import paths
import pandas as pd #libreria pandas para guardar las caracteristicas en excel/csv.
import argparse
from imutils import paths
from descriptores.color import MomentosColor
from sklearn.preprocessing import MinMaxScaler
#Obtener argumentos desde consola con Argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Ruta del Dataset de Imagenes Segmentadas..")
ap.add_argument("-o", "--salida", required=True, help="Ruta de la carpeta para guardar los vectores de características.")
args = vars(ap.parse_args())


dataset_imagenes = args["dataset"]

#Variables que contienen la ruta para guardar el vector
#de características y etiquetas/clases de limón sutil.
salida_data = args["salida"]+"/data.h5"
salida_labels = args["salida"]+"/labels.h5"

#Listas para guardar las características y clases de limón sutil.
caracteristicas = []
labels = []


#Obtener la ruta de las imagenes como lista:
imagenRutas = list(paths.list_images(dataset_imagenes))

t = len(imagenRutas)
#Iterar de acuerdo al numero de imagenes en el dataset:
actualizaciones = 100
for (i, imagenRutas) in enumerate(imagenRutas):
    #Leer imagen
    image = cv2.imread(imagenRutas)
    #Obtener nombre de etiqueta/clase de la imagen actual
    #se obtiene del nombre de la carpeta que contiene x imagen.
    label = imagenRutas.split(os.path.sep)[-2]

    #Obtener característica de Momentos de Color
    momcolor = MomentosColor(image)
    crtmoncolor = momcolor.getImageMoments()
    #Poner las caracteristicas y clase en las listas:
    labels.append(label)
    caracteristicas.append(crtmoncolor)

    if actualizaciones > 0 and i > 0 and (i+1)%actualizaciones == 0:
        print("[INFO] Procesado.. {}/{}".format(i+1, t))

print("[INFO] Características extraidas...")
print("[INFO] Tamaño de vector de caracteristicas: {}".format(np.array(caracteristicas).shape))
print("[INFO] Categorias/labels para entrenamiento obtenidas...{}".format(np.array(labels).shape))
print ("[INFO] Codificando las etiquetas/labels mediante ONE-HOT Encode.")

# ONE-HOT encode para las etiquetas
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)


print("[INFO] Guardando características en excel...")
# Convertir datos del array de caracteristicas en dataframe
df = pd.DataFrame(caracteristicas)
filepath = args["salida"]+str('/caracteristicas_momentoscolor.xlsx')
df.to_excel(filepath, index=False)


print("[INFO] Normalizando Características...")
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
estimator = Lasso()
featureSelection = SelectFromModel(estimator)
featureSelection.fit(caracteristicas, target)
selectedFeatures = featureSelection.transform(caracteristicas)
"""
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(caracteristicas)
print("[INFO] Vector de caracteristicas normalizado...")
print("[INFO] Guardando características de momentos de color normalizadas en excel...")

# Convertir datos del vectores de caracteristicas en dataframe
df = pd.DataFrame(rescaled_features)
filepath = args["salida"]+str('/caracteristicas_momentoscolor_normalizadas.xlsx')
df.to_excel(filepath, index=False)


print("[INFO] Target labels: {}".format(target))
print("[INFO] Target labels shape: {}".format(target.shape))
print("[INFO] Guardando  caracteristicas en formato H5")

# Guardar vector de caracteristicas utilizando formato H5
h5f_data = h5py.File(salida_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
h5f_label = h5py.File(salida_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))
h5f_data.close()
h5f_label.close()
print("[STATUS] terminado..")