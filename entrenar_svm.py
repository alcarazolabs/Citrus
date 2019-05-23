#Importar paquetes necesarios
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from imutils import paths
import cv2
import matplotlib.pyplot as plt
from descriptores.color import MomentosColor
import argparse
import pickle
import matplotlib.pyplot as plt


#Obtener argumentos desde consola
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Ruta para guardar el modelo")
args = vars(ap.parse_args())

# Importar vectores de características:
h5f_data = h5py.File('salida/data.h5', 'r')
h5f_label = h5py.File('salida/labels.h5', 'r')

data = h5f_data['dataset_1']
labels = h5f_label['dataset_1']

#Poner los datos en arrrays:
data = np.array(data)
labels = np.array(labels)

#Cerrar archivos hdf5.
h5f_data.close()
h5f_label.close()

#Dividir el conjunto de datos en entrenamiento y prueba:
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

print(x_train)

#Crear labels
labels = ["Maduro", "Pinton", "Verde"]
#Codificar las labels en ONE-HOT encode:
le = LabelEncoder()
labels = le.fit_transform(labels)

#Crear clasificador SVM
svm = SVC(kernel='rbf')
#Entrenar SVM:
svm.fit(x_train, y_train)
#Mostrar Resultados:
print(classification_report(y_test, svm.predict(x_test), target_names=le.classes_))


#Guardar Pesos
with open('svm.pkl', 'wb') as f:
    pickle.dump(svm, f)

print("Probando Clasificador SVM")

# path to test data
test_ruta = "dataset/test/"
# loop through the test images
imagenRutas = list(paths.list_images(test_ruta))

total = len(imagenRutas)
classLabels = ["maduro", "pinton", "verde"]

# loop over the input images
for (i, imagenRutas) in enumerate(imagenRutas):
    image = cv2.imread(imagenRutas)
    #obtener caracteristicas
    imgmoments = MomentosColor(image)
    features = imgmoments.getImageMoments()
    x = np.array(features).reshape(1,-1)
    #Predecir la clase para la imágen actual
    prediction = svm.predict(x)
    # Mostrar Etiqueta Predecida Sobre una Imagen de Prueba
    cv2.putText(image, classLabels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # Mostrar la imágen de salida
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
