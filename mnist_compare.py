from __future__ import print_function
from __future__ import division
import keras
import tensorflow as tf
import numpy as np
from numpy.random import seed
import os
import random

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import Callback

batch_size = 256
num_classes = 10
epochs = 1
trials = 10

# input image dimensions
img_rows, img_cols = 28, 28

class LossHistory(Callback):
    def __init__(self):
        self.iteration = 0
        self.losses = [[] for i in range(trials)]  

    def on_batch_end(self, batch, logs={}):
      self.losses[self.iteration].append(logs.get('loss'))

class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights('weights/' + name)
        self.batch += 1

#model = load_model('6-13-18-simpleMNIST.h5')


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def train():
  # Uncomment code below to generate same neural network.
  # seed(1)
  # tf.set_random_seed(1)
  # os.environ['PYTHONHASHSEED'] = '1'
  # random.seed(1)
  # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
  # keras.backend.set_session(sess)

  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[loss])
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

def squaredDifference(x, y):
  temp = 0
  for i in range(trials):
    temp += (x[i] - y[i])**2
  return temp

loss = LossHistory()
for i in range(trials):
  print('___________________')
  print('trial ' + str(i+1))
  loss.iteration = i
  train()
print(loss.losses)


comparison = [[squaredDifference(loss.losses[i], loss.losses[j]) for i in range(trials)] for j in range(trials)]  
#model.save('6-13-18-simpleMNISTv1.h5')