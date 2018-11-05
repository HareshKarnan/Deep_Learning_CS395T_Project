import keras.metrics
from keras.models import load_model
import keras.backend as K
from keras.applications import vgg16
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

model = load_model('/home/cs395t-f17/model/fitted_models/m_2018-10-08_00:57_VGGFace_categorical_crossentropy_adam_lr0.001_epochs50_regnone_decay2e-05_tl.model')
model.summary()
from skimage.io import imsave
img_path = '/home/yearbook/train/1905/M_006929.png'
# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor = vgg16.preprocess_input(img_tensor)

# Its shape is (1, 171, 186, 3)
print(img_tensor.shape)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.savefig('/home/tensorimage.jpg')
print('saved the image')
