from __future__ import print_function
from __future__ import division
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import Callback
import numpy as np

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 12, 12, 64)        0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 9216)              0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               1179776   
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 128)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1290      
# =================================================================
# Total params: 1,199,882
# Trainable params: 1,199,882
# Non-trainable params: 0
desired_batch = 0
weight_layer = 0

model = load_model('6-13-18-simpleMNISTv1.h5') #to get architecture
name = 'weights%08d.h5' % desired_batch
model.load_weights('weights/' + name, by_name = False)

#model.summary()
weights = model.get_weights()
#print(len(weights))

temp = weights[weight_layer]
#print(temp.shape)
#print(temp.T.shape)
u, s, vh = np.linalg.svd(temp.T) #transpose to get (kernel, channel, x, y)
print(u.shape)
print(s.shape)
print(vh.shape)
