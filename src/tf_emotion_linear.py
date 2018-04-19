import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

# None : the number of input images
# 48 x 48 x 1 , gray scale image
X = tf.placeholder('float', [None, 48*48])
W = tf.Variable(tf.zeros([48*48, 7]))
b = tf.Variable(tf.zeros([7]))
y = tf.matmul(X, W) +  b

# Define loss and optimizer
y_ = tf.placeholder('float', [None, 7])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train

for _ in range (10000):
    X_batch = None
    Y_batch = None

    shuffle_indexes = np.arange(len(X_train))
    np.random.shuffle(shuffle_indexes)
    shuffle_indexes = shuffle_indexes[0:100]
    X_batch = X_train[shuffle_indexes,:]
    Y_batch = Y_train[shuffle_indexes]

    sess.run(train_step, feed_dict={X: X_batch, y_: Y_batch})

    #Test

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('accuracy is : %d' % _)
    print(sess.run(accuracy, feed_dict = {X: X_test, y_: Y_test}))


