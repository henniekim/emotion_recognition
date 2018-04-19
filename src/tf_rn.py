import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#######################################
#        Pre Defined Function
#######################################

# functions below are defined to use it simple

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#######################################
#             DATA LOAD
#######################################

X_train = np.load('./data/train_set_fer2013_vector.npy')
Y_train = np.load('./data/train_labels_fer2013_vector.npy')
X_test = np.load('./data/test_set_fer2013_vector.npy')
Y_test = np.load('./data/test_labels_fer2013_vector.npy')

X_train = X_train.reshape(X_train.shape[0], 48*48)
X_test= X_test.reshape(X_test.shape[0], 48*48)

print ('---------------------------------')
print (' Training Data : %d' % len(X_train))
print (' Test Data : %d' % len(X_test))
print ('---------------------------------')


#######################################
#             CONSTANT
#######################################

learning_rate = 0.0001
batch_size =256
epoch = 200

print ('---------------------------------')
print (' batch size : %d' % batch_size)
print (' epoch : %d' % epoch)
print (' learning rate : %f' % learning_rate)
print ('---------------------------------')

sizeofdata = len(X_train)

X = tf.placeholder(tf.float32, [None, 48*48])
x_image = tf.reshape(X, [-1, 48, 48, 1])



#######################################
# HIDDEN LAYER 1 #
# input : 4D matrix as follows [ the number of training sample x face size x face size x 1 ] : [ 13746 x 48 x 48 x 1 ]
# weight : 48 x 48 x 64 x 1
# output : [1 x 48 x 48 x 64 ]
#######################################

W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])
r_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding = 'SAME') + b_conv1
h_conv1 = tf.nn.relu(r_conv1)


#######################################
# MAX POOLING LAYER 1
# input : [1 x 48 x 48 x 64]
# output : [1 x 24 x 24 x 64]
#######################################

h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


#######################################
# HIDDEN LAYER 2
# input : [64 x 24 x 24 x 64]
# output : [64 x 24 x 24 x 64]
#######################################

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
r_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding = 'SAME') + b_conv2
h_conv2 = tf.nn.relu(r_conv2)


#######################################
# MAX POOLING LAYER 2
# input : [64 x 24 x 24 x 64]
# output : [64 x 12 x 12 x 64]
#######################################

h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#######################################
# fully connected LAYER
# input : [64 x 12 x 12 x 64]
# output : [12 x 12 x 3072]
#######################################

W_fc1 = weight_variable([12 * 12 * 64, 1024])
b_fc1 = bias_variable([1024])
h_conv2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

#######################################
# Drop Out LAYER
#######################################

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#######################################
# Fully Connected LAYER : Softmax
# input : [12*12*128 x 3072]
# output : [7 x 1]
#######################################

W_fc2 = weight_variable([1024, 7])
b_fc2 = bias_variable([7])
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

########################################
# Define loss and optimizer
# Loss : cross entropy
# Optimizer : Adam
########################################

y_ = tf.placeholder(tf.float32, [None, 7])

softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

########################################
# Accuracy
########################################

correct_prediction = tf.equal(tf.argmax(y,1 ), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

########################################
# Open Session
########################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()

test_acc = list()

########################################
# Training
########################################

for i in range(epoch):

    # Each epoch, shuffling all of the data set
    shuffled_indexes = np.arange(len(X_train))
    np.random.shuffle(shuffled_indexes)

    for j in range(len(X_train) / batch_size):

        # Make mini batch
        if j == (len(X_train) / batch_size):
            shuffle_indexes = shuffled_indexes[batch_size * j:len(X_train)]
            X_batch = X_train[shuffle_indexes, :]
            Y_batch = Y_train[shuffle_indexes]
        else:
            shuffle_indexes = shuffled_indexes[batch_size * j:batch_size * (j + 1)]
            X_batch = X_train[shuffle_indexes, :]
            Y_batch = Y_train[shuffle_indexes]
        train_step.run(feed_dict= {X: X_batch, y_: Y_batch, keep_prob : 0.7})

        ########################################
        # Add here to augment data
        ########################################

        train_accuracy = accuracy.eval(feed_dict = { X : X_batch, y_ : Y_batch, keep_prob : 1.0})
        #print(' training step %d / %d , training acc : %f' % (j*batch_size, len(X_train), train_accuracy))

    test_accuracy = accuracy.eval(feed_dict = { X : X_test, y_ : Y_test, keep_prob : 1.0 })
    test_acc.append([epoch, test_accuracy])
    print('epoch is %d / %d, test acc : %f' % (i, epoch, test_accuracy))

print ('Training has been done !')
save_path = saver.save(sess, "./training/emotion_model_reduced.ckpt")

print (' Model saved in file : %s' % save_path )


W0 = r_conv2[1,:,:,1]

W0 = W0.eval(feed_dict = {X: X_batch, y_: Y_batch})

#W0 = np.mean(W0, axis =2)

plt.imshow(W0, cmap = cm.gray)

print ('the end')

