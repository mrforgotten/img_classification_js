import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt


def predict_pizza(x):
    per = x[0]*100
    if per >= 94.0:
        return 'pizza'
    else:
        return 'not_pizza'

model = load_model('/content/drive/My Drive/model/modeloncolabv1epoch30')

import sys

img = load_img('insertimgurl',target_size=(256,256))
plt.imshow(img)
plt.show()
img_array = img_to_array(img)*(1./255.)
img_array = tf.expand_dims(img_array, 0)
predictor = model.predict(img_array)
score = predictor[0]
print(
    'It has pizza percent is {} and it is {}'.format(score*100, predict_pizza(score))
)
