# Preparation: make symbolic links for practice_train_10k.root and practice_test_10k.root
import subprocess
# ln -sf ../../mix.root ./train_pdecay.root
# ln -sf ../../mix.root ./test_pdecay.root

from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys,time

# tensorflow/gpu start-up configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf

TUTORIAL_DIR     = '..'
TEST_IO_CONFIG   = os.path.join(TUTORIAL_DIR, 'tf/pdecay_analysis.cfg' )
TEST_BATCH_SIZE  = 100
SNAPSHOT = "weights_pdecay/toynet-24999"
NUM_CLASS = 2

#
# Step 0: IO

# for "test" data set
test_io = larcv_threadio()   # create io interface
test_io_cfg = {'filler_name' : 'TestIO',
               'verbosity'   : 0,
               'filler_cfg'  : TEST_IO_CONFIG}
test_io.configure(test_io_cfg)   # configure
test_io.start_manager(TEST_BATCH_SIZE) # start read thread
time.sleep(2)
test_io.next(store_entries=True, store_event_ids=True)

#
# Step 1: Define network
#
import tensorflow.contrib.slim as slim
import tensorflow.python.platform

def build(input_tensor, num_class=4, trainable=True, debug=True):

    net = input_tensor
    if debug: print('input tensor:', input_tensor.shape)

    filters = 32
    num_modules = 5
    with tf.variable_scope('conv'):
        for step in xrange(5):
            stride = 2
            if step: stride = 1
            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = stride,     # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
                              normalizer_fn = slim.batch_norm,
            scope         = 'conv%da_conv' % step)

            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = 1,          # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
            normalizer_fn = slim.batch_norm,
                              scope         = 'conv%db_conv' % step)
            if (step+1) < num_modules:
                net = slim.max_pool2d(inputs      = net,    # input tensor
                                      kernel_size = [2,2],  # kernel size
                                      stride      = 2,      # stride size
                                      scope       = 'conv%d_pool' % step)

            else:
                net = tf.layers.average_pooling2d(inputs = net,
                                                  pool_size = [net.get_shape()[-2].value,net.get_shape()[-3].value],
                                                  strides = 1,
                                                  padding = 'valid',
                                                  name = 'conv%d_pool' % step)
            filters *= 2

            if debug: print('After step',step,'shape',net.shape)

    with tf.variable_scope('final'):
        net = slim.flatten(net, scope='flatten')

        if debug: print('After flattening', net.shape)

        net = slim.fully_connected(net, int(num_class), scope='final_fc')

        if debug: print('After final_fc', net.shape)

    return net

#
# Step 2: Build network + define loss & solver
#
# retrieve dimensions of data for network construction
dim_data  = test_io.fetch_data('test_image').dim()
# define place holders
data_tensor    = tf.placeholder(tf.float32, [None, dim_data[1] * dim_data[2] * dim_data[3]], name='image')
data_tensor_2d = tf.reshape(data_tensor, [-1,dim_data[1],dim_data[2],dim_data[3]],name='image_reshape')
import sys

# build net
net = build(input_tensor=data_tensor_2d, num_class=NUM_CLASS, trainable=False, debug=False)
softmax_op = tf.nn.softmax(logits=net)
#                                                                                                                                                                                                                                          
# Create a session                                                                                                                     
sess = tf.InteractiveSession()
tf.global_variables_initializer()

# Create weights saver                                                                                                                 
saver = tf.train.Saver()
saver.restore(sess, SNAPSHOT)

csv_filename = 'proton-decay-inference.csv'
fout=open(csv_filename,'w')
fout.write('entry,run,subrun,event,label,prediction,probability\n')
ctr = 0
num_events = test_io.fetch_n_entries()
while ctr < num_events:
    print(str(ctr))
    test_data  = test_io.fetch_data('test_image').data()
    test_label = test_io.fetch_data('test_label').data()
    feed_dict = { data_tensor  : test_data }

    softmax_batch     = sess.run(softmax_op, feed_dict=feed_dict)
    processed_events  = test_io.fetch_event_ids()
    processed_entries = test_io.fetch_entries()
  
    for j in xrange(len(softmax_batch)):
        softmax_array = softmax_batch[j]
        entry         = processed_entries[j]
        event_id      = processed_events[j]
        label = np.argmax(test_label[j])
        prediction      = np.argmax(softmax_array)
        prediction_prob = softmax_array[prediction]
        
        data_string = '%d,%d,%d,%d,%d,%d,%g\n' % (entry,event_id.run(),event_id.subrun(),event_id.event(), label, prediction, prediction_prob)
        fout.write(data_string)

        ctr += 1
        if ctr == num_events:
            break
    if ctr == num_events:
        break

    test_io.next(store_entries=True,store_event_ids=True)

test_io.reset()
fout.close()


import pandas as pd
df = pd.read_csv(csv_filename)
df.describe()