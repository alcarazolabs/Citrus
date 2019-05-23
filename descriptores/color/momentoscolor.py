import cv2
import numpy as np
from scipy.stats import skew


class MomentosColor(object):
	def __init__(self, image):
		self.image = image

	def getMean(self, r_filtered, g_filtered, b_filtered):

		#Calcular la media para los 3 canales:
		r_mean = np.mean(r_filtered, axis=(0, 1))
		g_mean = np.mean(g_filtered, axis=(0, 1))
		b_mean = np.mean(b_filtered, axis=(0, 1))
		#r_mean[--, --, mean]
		#g_mean[--, mean, --]
		#b_mean[mean, --, --]
		#Guardar el color medio para las componentes R,G y B
		rgb_mean = []
		rgb_mean = [r_mean[2],g_mean[1],b_mean[0]]
		mean = []
		mean.append(rgb_mean[0])
		mean.append(rgb_mean[1])
		mean.append(rgb_mean[2])
		return (np.array(mean))

	def getStd(self, r_filtered, g_filtered, b_filtered):
		#Calcular la desviación estandar de los 3 channels:
		r_std = np.std(r_filtered, ddof=1)
		g_std = np.std(g_filtered, ddof=1)
		b_std = np.std(b_filtered, ddof=1)
		#Guardar valores de los 3 canales en una array
		rgb_std = []
		rgb_std = [r_std,g_std,b_std]
		#Guardar valores en un segundo array para retornar
		std = []
		std.append(rgb_std[0])
		std.append(rgb_std[1])
		std.append(rgb_std[2])
		return (np.array(std))

	def getSkewness(self, r_filtered, g_filtered, b_filtered):
		#Convertir las matrices de los canales en arrays 1-D
		#R
		r_filtered = np.array(r_filtered)
		r_filtered = r_filtered.ravel()
		#remover ceros
		r_filtered = r_filtered[r_filtered != 0]
		#G
		g_filtered = np.array(g_filtered)
		g_filtered = g_filtered.ravel()
		#remover ceros
		g_filtered = g_filtered[g_filtered != 0]
		#B
		b_filtered = np.array(b_filtered)
		b_filtered = b_filtered.ravel()
		#remover ceros
		b_filtered = b_filtered[b_filtered != 0]
		#Calcular la Skewness para los 3 channels:
		r_skew = skew(r_filtered)
		g_skew = skew(g_filtered)
		b_skew = skew(b_filtered)
		skewness = []
		skewness.append(r_skew)
		skewness.append(g_skew)
		skewness.append(b_skew)
		return np.array(skewness)

	def getImageMoments(self):
		r_channel = self.image.copy()
		g_channel = self.image.copy()
		b_channel = self.image.copy()
		#Obtener el canal rojo
		r_channel[:, :, 0] = 0
		r_channel[:, :, 1] = 0
		#Obtener el canal verde
		g_channel[:, :, 0] = 0
		g_channel[:, :, 2] = 0
		#Obtener el canal azul
		b_channel[:, :, 1] = 0
		b_channel[:, :, 2] = 0
		#Obtener los canales como arreglos
		#R
		r_array = np.asarray(r_channel)
		#G
		g_array = np.asarray(g_channel)
		#B
		b_array = np.asarray(b_channel)
		#Crear mascara para omitir el uso de pixeles con valor 0
		#R
		r_filtered = np.ma.masked_where(r_array == 0, r_array)
		#G
		g_filtered = np.ma.masked_where(g_array == 0, g_array)
		#B
		b_filtered = np.ma.masked_where(b_array == 0, b_array)

		meann = self.getMean(r_filtered, g_filtered, b_filtered)
		std = self.getStd(r_filtered, g_filtered, b_filtered)
		skewness = self.getSkewness(r_filtered, g_filtered, b_filtered)

		#retornar vector de 9 momentos de color:
		return list(np.array(np.concatenate([meann, std, skewness])))