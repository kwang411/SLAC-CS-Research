from IPython import display
from IPython.display import HTML
import commands,sys,os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Download & extract mnist data set
from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets('MNIST_data', one_hot=True)

# Constants used in this notebook
MNIST_SIZE       = 28   # MNIST image size
MNIST_NUM_CLASS  = 10   # Number of classes in MNIST (digits 0 to 9)
IMAGE_SIZE       = 128  # Segmentation data image size we will create
TRAIN_DATA_SIZE  = 5000 # Number of images in the whole train data
TEST_DATA_SIZE   = 1000 # Number of images in the whole test  data
TRAIN_BATCH_SIZE = 20   # Batch size for training input 
TEST_BATCH_SIZE  = 10  # Batch size for testing  input

# Decorative progress bar
def progress(count, total, unit, message=''):
    return HTML("""
        <progress 
            value='{count}'
            max='{total}',
            style='width: 30%'
        >
            {count}
        </progress> {count}/{total} {unit} ({frac}%) ... {message}
    """.format(count=count, total=total, unit=unit, frac=int(float(count)/float(total)*100.),message=message))

# Create 1 segmentation data set ... image, label, and weights
def fill_mnist_segdata(big_image, big_label, big_weight, num_digits=6, train=True):
    source = MNIST.train
    if not train:
        source = MNIST.test
    
    image_size = big_image.shape[0]
    label = np.zeros([MNIST_SIZE]*2,dtype=np.float32)
    for _ in xrange(num_digits):
        index = np.random.randint(0,len(source.images))
        image = source.images[index]
        image = image.reshape([MNIST_SIZE]*2)

        label.fill(0)
        label_val = np.argmax(source.labels[index])+1
        label[np.where(image>0)] = label_val

        x_pos = int(np.random.randint(0,image_size-MNIST_SIZE))
        y_pos = int(np.random.randint(0,image_size-MNIST_SIZE))

        big_image[x_pos:x_pos+MNIST_SIZE,y_pos:y_pos+MNIST_SIZE] += image

        label_crop = big_label[x_pos:x_pos+MNIST_SIZE,y_pos:y_pos+MNIST_SIZE]
        label_crop[label>label_crop] = label_val

    vals, counts = np.unique(big_label,return_counts=True)
    for i, val in enumerate(vals):
        big_weight[np.where(big_label == val)] = 1. / counts[i] / len(vals)

# Create a whole segmentation data set for train/test
def make_mnist_segdata_big(data_size=5500,image_size=128,num_digits=6,train=True):
    print('Generating {data_size} images...'.format(data_size=data_size))  
    images  = np.zeros([data_size,image_size,image_size],dtype=np.float32)
    labels  = np.zeros([data_size,image_size,image_size],dtype=np.int64  )
    weights = np.zeros([data_size,image_size,image_size],dtype=np.float32)
    display.display(progress(0,data_size,'images'), display_id = True)
    for _ in range(data_size):
        image  = images  [_]
        label  = labels  [_]
        weight = weights [_]
        fill_mnist_segdata(image,label,weight,num_digits,train=train)
        display.clear_output(wait=True)
        display.display(progress(_,data_size,'images'))
    display.clear_output(wait=True)
    return images, labels, weights

# Generate data sets
train_images, train_labels, train_weights = make_mnist_segdata_big(TRAIN_DATA_SIZE, train=True)
test_images,  test_labels,  test_weights  = make_mnist_segdata_big(TEST_DATA_SIZE,  train=False)
print('done!')

# Visualize a random entry
index = np.random.randint(0,len(train_images))
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,12),facecolor='w')
ax1.imshow(train_images[index],  interpolation='none',cmap='gray')
ax2.imshow(train_labels[index],  interpolation='none',cmap='jet',vmin=0,vmax=10)
ax3.imshow(train_weights[index], interpolation='none',cmap='jet')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax1.set_title('Image',fontsize=20)
ax2.set_title('Label',fontsize=20)
ax3.set_title('Weights',fontsize=20)
plt.show()

sess = tf.InteractiveSession()
with tf.variable_scope("train_input"):
    #Create a dataset tensor from the images and the labels
    train_dataset  = tf.data.Dataset.from_tensor_slices((train_images,train_labels,train_weights))
    train_dataset  = train_dataset.shuffle(buffer_size=5500,reshuffle_each_iteration=True).batch(TRAIN_BATCH_SIZE)
    train_iterator = train_dataset.make_initializable_iterator()
    # It is better to use 2 placeholders, to avoid to load all data into memory,
    # and avoid the 2Gb restriction length of a tensor.
    _batch_train_images  = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE])
    _batch_train_labels  = tf.placeholder(tf.int64,   [None, IMAGE_SIZE, IMAGE_SIZE])
    _batch_train_weights = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE])
    # Initialize the iterator
    sess.run(train_iterator.initializer, 
             feed_dict={_batch_train_images : train_images,
                        _batch_train_labels : train_labels,
                        _batch_train_weights: train_weights})

    # Neural Net Input
    batch_train_images, batch_train_labels, batch_train_weights = train_iterator.get_next()
    batch_train_images = tf.reshape(batch_train_images, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    batch_train_labels = tf.reshape(batch_train_labels, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

tf.summary.image  ('image_example', batch_train_images,3)
tf.summary.image  ('label_example', tf.image.grayscale_to_rgb(tf.cast(batch_train_labels,tf.float32),'gray_to_rgb'),3)
batch_train_labels = tf.reshape(batch_train_labels, [-1, IMAGE_SIZE, IMAGE_SIZE])

with tf.variable_scope("test_input"):
    #Create a dataset tensor from the images and the labels
    test_dataset  = tf.data.Dataset.from_tensor_slices((test_images,test_labels,test_weights))
    test_dataset  = test_dataset.shuffle(buffer_size=1000,reshuffle_each_iteration=True).batch(TEST_BATCH_SIZE)
    test_iterator = test_dataset.make_initializable_iterator()
    # It is better to use 2 placeholders, to avoid to load all data into memory,
    # and avoid the 2Gb restriction length of a tensor.
    _batch_test_images  = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE])
    _batch_test_labels  = tf.placeholder(tf.int64,   [None, IMAGE_SIZE, IMAGE_SIZE])
    _batch_test_weights = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE])
    # Initialize the iterator
    sess.run(test_iterator.initializer, 
             feed_dict={_batch_test_images:  test_images,
                        _batch_test_labels:  test_labels,
                        _batch_test_weights: test_weights})

    # Neural Net Input
    batch_test_images,  batch_test_labels, batch_test_weights = test_iterator.get_next()
    batch_test_images  = tf.reshape(batch_test_images,  [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

import tensorflow.python.platform
import tensorflow.contrib.layers as L
import tensorflow.contrib.slim as slim

def toy_resnet_module(input_tensor, num_outputs, trainable=True, kernel=(3,3), stride=1, scope='noscope'):

    fn_conv = slim.conv2d

    num_inputs  = input_tensor.get_shape()[-1].value
    with tf.variable_scope(scope):
        #
        # shortcut path
        #
        shortcut = None
        if num_outputs == num_inputs and stride ==1 :
            shortcut = input_tensor
        else:
            shortcut = slim.conv2d(inputs      = input_tensor,
                                   num_outputs = num_outputs,
                                   kernel_size = 1,
                                   stride      = stride,
                                   normalizer_fn = None, #slim.batch_norm
                                   activation_fn = None,
                                   trainable   = trainable,
                                   scope       = 'shortcut')
        #
        # residual path
        #
        residual = slim.conv2d(inputs      = input_tensor,
                               num_outputs = num_outputs,
                               kernel_size = kernel,
                               stride      = stride,
                               normalizer_fn = None, #slim.batch_norm
                               #activation_fn = None,
                               trainable   = trainable,
                               scope       = 'resnet_conv1')
        
        residual = slim.conv2d(inputs      = residual,
                               num_outputs = num_outputs,
                               kernel_size = kernel,
                               normalizer_fn = None, #slim.batch_norm
                               activation_fn = None,
                               trainable   = trainable,
                               scope       = 'resnet_conv2')
        
        return tf.nn.relu(shortcut + residual)

def double_toy_resnet(input_tensor, num_outputs, trainable=True, kernel=3, stride=1, scope='noscope'):

    with tf.variable_scope(scope):

        resnet1 = toy_resnet_module(input_tensor=input_tensor,
                                    trainable=trainable,
                                    kernel=kernel,
                                    stride=stride,
                                    num_outputs=num_outputs,
                                    scope='module1')

        resnet2 = toy_resnet_module(input_tensor=resnet1,
                                    trainable=trainable,
                                    kernel=kernel,
                                    stride=1,
                                    num_outputs=num_outputs,
                                    scope='module2')

        return resnet2

def toy_uresnet(input_tensor, num_class, reuse=False, trainable=True, base_filter=32, num_contraction=4):

    with tf.variable_scope('toy_uresnet', reuse=reuse):

        conv_feature_map={}
        net = input_tensor
        print('Input shape {:s}'.format(net.shape))

        # 1st conv layer normal
        net = slim.conv2d     (net, base_filter, 3, normalizer_fn= None, trainable=trainable, scope='conv0') # slim.batch_norm
        conv_feature_map[net.get_shape()[-1].value] = net 
        print('Encoding step 0 shape {:s}'.format(net.shape))  

        net = slim.max_pool2d (net,              2, scope='maxpool0')    
        # encoding steps
        for step in range(num_contraction):
            num_outputs = base_filter * (2**(step+1))
            stride = 2
            if step == 0: stride = 1
            net = double_toy_resnet(net, num_outputs, trainable=trainable, stride=stride, scope='res{:d}'.format(step+1))
            conv_feature_map[net.get_shape()[-1].value] = net
            print('Encoding step {:d} shape {:s}'.format(step+1,net.shape))
        # decoding steps
        for step in range(num_contraction):
            num_outputs = net.get_shape()[-1].value / 2
            net = slim.conv2d_transpose(net, num_outputs, 3, stride=2, normalizer_fn=None, trainable=trainable, scope='deconv{:d}'.format(step)) # slim.batch_norm
            net = tf.concat([net, conv_feature_map[num_outputs]], axis=len(net.shape)-1, name='concat{:d}'.format(step))
            net = double_toy_resnet(net, num_outputs, trainable=trainable, scope='conv{:d}'.format(step+num_contraction+1))
            print('Decoding {:d} shape {:s}'.format(step,net.shape))

        # final conv layer
        net = slim.conv2d(net, num_class, 3, normalizer_fn=None, trainable=trainable, scope='lastconv') #slim.batch_norm
        print('Final shape {:s}'.format(net.shape))  
        return net

print('Building train net...')
train_net = toy_uresnet (batch_train_images, MNIST_NUM_CLASS+1, trainable=True,  reuse=False )

print('\nBuilding test net...')
test_net  = toy_uresnet (batch_test_images,  MNIST_NUM_CLASS+1, trainable=False, reuse=True  )

with tf.variable_scope('analysis'):
    prediction     = tf.argmax(test_net,3)
    accuracy_allpx = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_test_labels),tf.float32))
    nonzero_idx    = tf.where(tf.reshape(batch_test_images, [-1, IMAGE_SIZE, IMAGE_SIZE]) > tf.to_float(0.) )
    nonzero_label  = tf.gather_nd(batch_test_labels, nonzero_idx)
    nonzero_pred   = tf.gather_nd(tf.argmax(test_net, 3), nonzero_idx)
    accuracy_valpx = tf.reduce_mean(tf.cast(tf.equal(nonzero_label, nonzero_pred),tf.float32))
    softmax        = tf.nn.softmax(logits=test_net)

with tf.variable_scope('train'):
    loss_pixel     = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_train_labels, logits=train_net)
    loss_weighted  = tf.multiply(loss_pixel, batch_train_weights)
    loss           = tf.reduce_mean(tf.reduce_sum(tf.reshape(loss_weighted, [-1, int(IMAGE_SIZE**2)]),axis=1))

    learning_rate  = tf.placeholder(tf.float32,[])
    train          = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize everything
sess.run(tf.global_variables_initializer())

class train_log:
    pass
log = train_log()
log.train_loss   = []
log.test_acc_val = []
log.test_acc_all = []
log.test_steps   = []

def imshow_test(num_images=None):
    ops = [batch_test_images,batch_test_labels,prediction,softmax]
    try:
        images, labels, preds, probs = sess.run(ops)
    except tf.errors.OutOfRangeError:

        # Reload the iterator when it reaches the end of the dataset
        sess.run(test_iterator.initializer,
                 feed_dict={_batch_test_images: test_images,
                            _batch_test_labels: test_labels})
        images, labels, preds, probs = sess.run(ops)
    if num_images is None or num_images > len(images):
        num_images = len(images)
    for index in range(num_images):
        fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,12),facecolor='w')
        ax1.imshow(images[index].reshape([IMAGE_SIZE,IMAGE_SIZE]), interpolation='none',cmap='gray')
        ax2.imshow(labels[index].reshape([IMAGE_SIZE,IMAGE_SIZE]), interpolation='none',cmap='jet',vmin=0,vmax=10)
        ax3.imshow(preds[index].reshape([IMAGE_SIZE,IMAGE_SIZE]),  interpolation='none',cmap='jet',vmin=0,vmax=10)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_title('Image',fontsize=20)
        ax2.set_title('Label',fontsize=20)
        ax3.set_title('Prediction',fontsize=20)
        plt.show()
        
def plot_log(log):
    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    ax1.plot(np.arange(0,len(log.train_loss)),log.train_loss,
                       linewidth=2,
                       label='Loss',color='b')
    ax1.set_xlabel('Iterations',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    ax1.set_ylim(0., 4.0)
    
    ax2 = ax1.twinx()
    ax2.plot(log.test_steps,1.- np.array(log.test_acc_val),
             marker='o',linestyle='',
             color='orange',label='Acc. Nonzero Px')
    ax2.plot(log.test_steps,1.- np.array(log.test_acc_all),color='magenta',label='Acc. All px')
    ax2.set_ylabel('Error Rate', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,0.2)

    plt.grid()
    plt.show()

imshow_test(1)

# Training
def train_loop(num_steps,log,lr=0.001,display_cycle=20):
    
    previous_steps = len(log.train_loss)
    print('Running train iterations {:d} => {:d}!'.format(previous_steps,previous_steps+num_steps))
    
    log.train_loss = np.append(log.train_loss, np.zeros([num_steps],dtype=np.float32))
    epoch_in_steps = float(TRAIN_DATA_SIZE) / float(TRAIN_BATCH_SIZE)
    step_in_epochs = 1./epoch_in_steps
    current_epoch  = int(previous_steps / epoch_in_steps) - 1
    display.display(progress(previous_steps,previous_steps+num_steps,'steps'),display_id=True)
    #epoch_report   = display.display(progress(0, epoch_in_steps, 'steps'),display_id=True)
    for step in range(num_steps):

        global_step = step + previous_steps

        try:
            # Run optimization
            _,log.train_loss[global_step] = sess.run([train,loss],feed_dict={learning_rate: lr})
        except tf.errors.OutOfRangeError:
            # Reload the iterator when it reaches the end of the dataset
            sess.run(train_iterator.initializer,
                     feed_dict = {_batch_train_images: train_images,
                                  _batch_train_labels: train_labels})
            _,log.train_loss[global_step] = sess.run([train,loss],feed_dict={learning_rate: lr})

        steps_this_epoch = int((global_step+1) - epoch_in_steps * current_epoch)
        if int((global_step+1) * step_in_epochs) > current_epoch:
            # imshow test @ epoch boundary    
            current_epoch += 1
            if current_epoch%2 == 0:
                imshow_test(1)
            #imshow_test(1)
            #report = display(progress(steps_this_epoch,int(epoch_in_steps),'steps'),display_id=True)

        message = 'Epochs {:d}/{:.2f}'.format(current_epoch,(previous_steps+num_steps)/epoch_in_steps)
        display.clear_output(wait=True)
        display.display(progress(global_step,previous_steps+num_steps,'steps',message))
        if len(log.test_acc_val)<1:
            message = 'Epoch {:d} loss {:.4f} '.format(current_epoch,log.train_loss[global_step])
            #display.display(progress(steps_this_epoch,int(epoch_in_steps),'steps',message))
        else:
            message = 'Epoch {:d} loss {:.4f} ... last test accuracy {:.3f} ... {:.3f}'
            message = message.format(current_epoch,
                                     log.train_loss[global_step],log.test_acc_all[-1],log.test_acc_val[-1])
            #display.display(progress(steps_this_epoch,int(epoch_in_steps),'steps',message))
        if (global_step+1) % display_cycle == 0:
            # Calculate batch loss and accuracy
            # (note that this consume a new batch of data)
            try:
                acc_val, acc_all = sess.run([accuracy_valpx, accuracy_allpx])
            except tf.errors.OutOfRangeError:
                # Reload the iterator when it reaches the end of the dataset
                sess.run(test_iterator.initializer,
                         feed_dict={_batch_test_images: test_images,
                                    _batch_test_labels: test_labels})
                acc_val, acc_all = sess.run([accuracy_valpx, accuracy_allpx])
            log.test_acc_val.append(acc_val)
            log.test_acc_all.append(acc_all)
            log.test_steps.append(global_step+1)

train_loop(10000,log,lr=0.001)

plot_log(log)

from tensorflow.python.client import device_lib

saver = tf.train.Saver()
#saver.save(sess, "semseg/nobatchnorm4")
