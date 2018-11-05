from os import path
import util
import numpy as np
import argparse
from skimage.io import imread, imsave
from util import *
import csv
import keras.metrics
from keras.models import load_model
import keras.backend as K
from keras.applications import vgg16
from keras.preprocessing import image

def mean_L1_distance(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def mean_L1_distance(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def min_L1_distance(y_true, y_pred):
    return K.min(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def max_L1_distance(y_true, y_pred):
    return K.max(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

def std_L1_distance(y_true, y_pred):
    return K.std(K.abs(K.argmax(y_pred,axis = -1) - K.argmax(y_true,axis = -1)), axis=-1)

keras.metrics.min_L1_distance= min_L1_distance
keras.metrics.max_L1_distance= max_L1_distance
keras.metrics.mean_L1_distance= mean_L1_distance

model = load_model('model/m_2018-10-08_00:57_VGGFace_categorical_crossentropy_adam_lr0.001_epochs50_regnone_decay2e-05_tl.model')
model.summary()

mapping = {
    0: 1905, 1: 1906, 2: 1908, 3: 1909, 4: 1910, 5: 1911, 6: 1912, 7: 1913, 8: 1914, 9: 1915,
    10: 1916, 11: 1919, 12: 1922, 13: 1923, 14: 1924, 15: 1925, 16: 1926, 17: 1927, 18: 1928,
    19: 1929, 20: 1930, 21: 1931, 22: 1932, 23: 1933, 24: 1934, 25: 1935, 26: 1936, 27: 1937,
    28: 1938, 29: 1939, 30: 1940, 31: 1941, 32: 1942, 33: 1943, 34: 1944, 35: 1945, 36: 1946,
    37: 1947, 38: 1948, 39: 1949, 40: 1950, 41: 1951, 42: 1952, 43: 1953, 44: 1954, 45: 1955,
    46: 1956, 47: 1957, 48: 1958, 49: 1959, 50: 1960, 51: 1961, 52: 1962, 53: 1963, 54: 1964,
    55: 1965, 56: 1966, 57: 1967, 58: 1968, 59: 1969, 60: 1970, 61: 1971, 62: 1972, 63: 1973,
    64: 1974, 65: 1975, 66: 1976, 67: 1977, 68: 1978, 69: 1979, 70: 1980, 71: 1981, 72: 1982,
    73: 1983, 74: 1984, 75: 1985, 76: 1986, 77: 1987, 78: 1988, 79: 1989, 80: 1990, 81: 1991,
    82: 1992, 83: 1993, 84: 1994, 85: 1995, 86: 1996, 87: 1997, 88: 1998, 89: 1999, 90: 2000,
    91: 2001, 92: 2002, 93: 2003, 94: 2004, 95: 2005, 96: 2006, 97: 2007, 98: 2008, 99: 2009,
    100: 2010, 101: 2011, 102: 2012, 103: 2013}





def load(img_path):
	#TODO:load image and process if you want to do any
	# We preprocess the image into a 4D tensor
	img = image.load_img(img_path, target_size=(224, 224))
	img_tensor = image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	# Remember that the model was trained on inputs
	# that were preprocessed in the following way:
	img_tensor = vgg16.preprocess_input(img_tensor)
	#img=imread(image_path)
	return img_tensor
	
class Predictor:
	DATASET_TYPE = 'yearbook'
	# baseline 1 which calculates the median of the train data and return each time
	def yearbook_baseline(self):
		# Load all training data
		train_list = listYearbook(train=True, valid=False)

		# Get all the labels
		years = np.array([float(y[1]) for y in train_list])
		med = np.median(years, axis=0)
		return [med]

	# Compute the median.
	# We do this in the projective space of the map instead of longitude/latitude,
	# as France is almost flat and euclidean distances in the projective space are
	# close enough to spherical distances.
	def streetview_baseline(self):
		# Load all training data
		train_list = listStreetView(train=True, valid=False)

		# Get all the labels
		coord = np.array([(float(y[1]), float(y[2])) for y in train_list])
		xy = coordinateToXY(coord)
		med = np.median(xy, axis=0, keepdims=True)
		med_coord = np.squeeze(XYToCoordinate(med))
		return med_coord

	def predict(self, image_path):
		img = load(image_path)
		#model = load_model('../model/m_2018-10-08_00:57_VGGFace_categorical_crossentropy_adam_lr0.001_epochs50_regnone_decay2e-05_tl.model')
		#model.summary()

		#TODO: load model

		#TODO: predict model and return result either in geolocation format or yearbook format
		# depending on the dataset you are using
		if self.DATASET_TYPE == 'geolocation':
			result = self.streetview_baseline() #for geolocation
		elif self.DATASET_TYPE == 'yearbook':
			#result = self.yearbook_baseline() #for yearbook
			pred_probs = model.predict(img)
			prediction_label = np.argmax(pred_probs)
			prediction_year = mapping[prediction_label]

		return [prediction_year]
	


