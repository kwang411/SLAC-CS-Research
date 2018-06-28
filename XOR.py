from __future__ import print_function
from __future__ import division
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
from numpy.random import seed
import os
import random
import tensorflow as tf

num = 0
seed(num)
tf.set_random_seed(num)
os.environ['PYTHONHASHSEED'] = str(num)
random.seed(num)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)



xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(units = 4, activation = 'tanh', name = 'input'))
model.add(Dense(units = 4, activation = 'tanh', name = 'hidden'))
model.add(Dense(units = 1, activation = 'sigmoid', name='output'))

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
model.save_weights('weightsLargeXOR/weights.%02d.00000.hdf5' % num)
model.fit(xor, y_xor, epochs=10000, callbacks= [ModelCheckpoint('weightsLargeXOR/weights.%02d.{epoch:05d}.hdf5' % num , monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=500)])


print("predicting [1, 0]: ")
print(model.predict_classes(np.array([[1, 0]])))
print("Predicting [0, 1]:")
print(model.predict_classes(np.array([[0, 1]])))
print("Predicting [0, 0]:")
print(model.predict_classes(np.array([[0, 0]])))
print("Predicting [1, 1]:")
print(model.predict_classes(np.array([[1, 1]])))

# model.save('6-28-18-largeXORv1.h5')
# del model
# model = load_model('6-25-18-simpleXORv1.h5') #to get architecture

#model.save('6-13-18-simpleMNISTv1.h5')