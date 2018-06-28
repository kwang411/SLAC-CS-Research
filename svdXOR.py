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
from numpy.random import seed
import os
import random
import tensorflow as tf
from keras.models import model_from_json

# json_string = '{"class_name": "Sequential", "keras_version": "2.2.0", "config": [{"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "input", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "activation": "tanh", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 4, "use_bias": true, "activity_regularizer": null}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "output", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "activation":"sigmoid", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 1, "use_bias": true, "activity_regularizer": null}}], "backend": "tensorflow"}'
# model = model_from_json(json_string)
print("_______________________________")
for num in [0,3,5,9]:

	print("random seed = " + str(num))
	epoch = 20000
	if num == 5:
		epoch = 25000 # that trial needed more time
	print("epoch = " + str(epoch))
	weight_layer = 0 #note that bias layers are in between weight layers

	model = load_model('6-25-18-simpleXORv1.h5') #to get architecture
	name = 'weightsXOR/weights.%02d.%05d.hdf5' % (num, epoch)
	model.load_weights(name, by_name = False)

	#model.summary()
	weights = model.get_weights()
	#print("len(weights) = " + str(len(weights)))
	#print("weights = " + str(weights))

	temp = weights[weight_layer]
	print("weight layer " + str(weight_layer) + " = " + str(temp))
	#print("temp.shape = " + str(temp.shape))
	#print("temp.T.shape = " + str(temp.T.shape))
	u, s, vh = np.linalg.svd(temp) #transpose to get (kernel, channel, x, y)
	#print("u.shape = " + str(u.shape))
	#print("s.shape = " + str(s.shape))
	#print("vh.shape = " + str(vh.shape))
	print("u = " + str(u))
	print("s = " + str(s))
	print("_______________________________")