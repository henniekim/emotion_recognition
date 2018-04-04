from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import tempfile
import sys

######################################
#            CONSTANT                #
######################################

FACE_SIZE = 48
BATCH_SIZE = 256
EPOCH = 100

#######################################
#             DATA LOAD               #
#######################################

X_train = np.load('./data/train_set_fer2013_vector.npy')
Y_train = np.load('./data/train_labels_fer2013_vector.npy')
X_test = np.load('./data/test_set_fer2013_vector.npy')
Y_test = np.load('./data/test_labels_fer2013_vector.npy')

X_train = X_train.reshape(X_train.shape[0], 48*48)
X_test= X_test.reshape(X_test.shape[0], 48*48)

Xtr_cols = X_train.reshape(48*48, X_train.shape[0])
Xte_cols = X_test.reshape(48*48, X_test.shape[0])
#######################################
#             INITIALIZE              #
#######################################

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 48, 48, 1])
        x_image = tf.cast(x_image, tf.float32 )

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5,5,1,64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1 )

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 64, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([12*12*64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 7])
        b_fc2 = bias_variable([7])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,3,3,1],
                          strides = [1,2,2,1], padding = 'SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main(_):
    x = tf.placeholder(tf.float32, [None, 48*48])

    y_ = tf.placeholder(tf.float32, [None, 7])

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            for i in range(len(X_train)):
    
                X_batch = None
                Y_batch = None
    
                shuffle_indexes = np.arange(len(X_train))
                np.random.shuffle(shuffle_indexes)
                shuffle_indexes = shuffle_indexes[0:256]
                X_batch = X_train[shuffle_indexes, :]
                Y_batch = Y_train[shuffle_indexes]
    
                if i % 200 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: X_batch, y_: Y_batch, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x: X_batch, y_: Y_batch, keep_prob: 0.3})
    
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: X_test, y_: Y_test, keep_prob: 1.0}))


tf.app.run(main=main, argv=[sys.argv[0]])

# angry template
'''
W0 = W[:,0]
W0 = tf.reshape(W0, [48,48])
W0 = W0.eval()
plt.imshow(W0, cmap = cm.gray)
cv2.waitKey()

# disgust template
W1 = W[:,1]
W1 = tf.reshape(W1, [48,48])
W1 = W1.eval()
plt.imshow(W1, cmap = cm.gray)
cv2.waitKey()

# fearful template
W2 = W[:,2]
W2 = tf.reshape(W2, [48,48])
W2 = W2.eval()
plt.imshow(W2, cmap = cm.gray)
cv2.waitKey()

# happy template
W3 = W[:,3]
W3 = tf.reshape(W3, [48,48])
W3 = W3.eval()
plt.imshow(W3, cmap = cm.gray)
cv2.waitKey(1)

# sad template
W4 = W[:,4]
W4 = tf.reshape(W4, [48,48])
W4 = W4.eval()
plt.imshow(W4, cmap = cm.gray)
cv2.waitKey(1)

# surprised template
W5 = W[:,5]
W5 = tf.reshape(W5, [48,48])
W5 = W5.eval()
plt.imshow(W5, cmap = cm.gray)
cv2.waitKey(1)

# neutral template
W6 = W[:,6]
W6 = tf.reshape(W6, [48,48])
W6 = W6.eval()
plt.imshow(W6, cmap = cm.gray)
cv2.waitKey(1)
'''
print ('the end')