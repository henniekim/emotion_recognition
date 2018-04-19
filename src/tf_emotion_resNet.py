# Smaller AlexNet : Hidden Layer 3

# Accuracy Data
# Batch 256, Epoch 500, Weight 0.01 , Bias 0.01 -> 65.76 % // 2018.01.10

import tensorflow as tf
import numpy as np
import io # for file in / out
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


debug_session = False

#######################################
#        Pre Defined Function         #
#######################################

# functions below are defined to use it simple

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02 )
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)

def make_hidden_patch(a, fill):
    for y in range(0,4):
        for x in range(0,4):
            key = np.random.randint(3, size = 1) # 50 %
            if key == 0 :
                for j in range(0,12) :
                    for i in range(0,12) :
                        a[12 * y * 48 + 12 * x+ j*48+i] = fill
    return a


#######################################
#             DATA LOAD               #
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
#             INITIALIZE              #
#######################################

learning_rate = 0.0001
batch_size = 256
epoch = 1000

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
# output : [1 x 48 x 48 x 64 ]
#######################################

W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])
r_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding = 'SAME') + b_conv1
h_conv1 = tf.nn.relu(r_conv1)

#######################################
# HIDDEN LAYER 2 #
# input : [64 x 48 x 48 x 64 ]
# output : [64 x 48 x 48 x 64 ]
#######################################

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
r_conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding = 'SAME') + b_conv2
h_conv2 = tf.nn.relu(r_conv2)


#######################################
# HIDDEN LAYER 3 #
# input : [64 x 48 x 48 x 64 ]
# output : [64 x 48 x 48 x 64 ]
#######################################

W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])
r_conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1,1,1,1], padding = 'SAME') + b_conv3
h_conv3 = tf.nn.relu(r_conv3)

#######################################
# HIDDEN LAYER 4 # (Residual  # 1)
# input : [64 x 48 x 48 x 64 ]
# output : [64 x 48 x 48 x 64 ]
#######################################

res4 = tf.add(h_conv1, h_conv3)
W_conv4 = weight_variable([5, 5, 64, 64])
b_conv4 = bias_variable([64])
r_conv4 = tf.nn.conv2d(res4, W_conv4, strides=[1,1,1,1], padding = 'SAME') + b_conv4
h_conv4 = tf.nn.relu(r_conv4)

#######################################
# HIDDEN LAYER 5 #
# input : [64 x 48 x 48 x 64 ]
# output : [64 x 48 x 48 x 64 ]
#######################################

W_conv5 = weight_variable([5, 5, 64, 64])
b_conv5 = bias_variable([64])
r_conv5 = tf.nn.conv2d(h_conv4, W_conv5, strides=[1,1,1,1], padding = 'SAME') + b_conv5
h_conv5 = tf.nn.relu(r_conv5)


#######################################
# HIDDEN LAYER 6 #
# input : [64 x 48 x 48 x 64 ]
# output : [64 x 48 x 48 x 64 ]
#######################################

W_conv6 = weight_variable([5, 5, 64, 64])
b_conv6 = bias_variable([64])
r_conv6 = tf.nn.conv2d(h_conv5, W_conv6, strides=[1,1,1,1], padding = 'SAME') + b_conv6
h_conv6 = tf.nn.relu(r_conv6)

#######################################
# HIDDEN LAYER 7 # (Residual # 2 )
# input : [64 x 48 x 48 x 64 ]
# output : [64 x 48 x 48 x 64 ]
#######################################

res7 = tf.add(h_conv4, h_conv6)
W_conv7 = weight_variable([5, 5, 64, 64])
b_conv7 = bias_variable([64])
r_conv7 = tf.nn.conv2d(res7, W_conv7, strides=[1,1,1,1], padding = 'SAME') + b_conv7
h_conv7 = tf.nn.relu(r_conv7)

#######################################
# MAX POOLING LAYER 1
# input : [64 x 48 x 48 x 64]
# output : [64 x 24 x 24 x 64]
#######################################

h_pool1 = tf.nn.max_pool(h_conv7, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

#######################################
# HIDDEN LAYER 8 #
# input : [64 x 24 x 24 x 64 ]
# output : [64 x 24 x 24 x 64 ]
#######################################

W_conv8 = weight_variable([3, 3, 64, 64])
b_conv8 = bias_variable([64])
r_conv8 = tf.nn.conv2d(h_pool1, W_conv8, strides=[1,1,1,1], padding = 'SAME') + b_conv8
h_conv8 = tf.nn.relu(r_conv8)


#######################################
# HIDDEN LAYER 9 #
# input : [64 x 24 x 24 x 64 ]
# output : [64 x 24 x 24 x 64 ]
#######################################

W_conv9 = weight_variable([3, 3, 64, 64])
b_conv9 = bias_variable([64])
r_conv9 = tf.nn.conv2d(h_conv8, W_conv9, strides=[1,1,1,1], padding = 'SAME') + b_conv9
h_conv9 = tf.nn.relu(r_conv9)

#######################################
# HIDDEN LAYER 10 # (Residual  # 3)
# input : [64 x 24 x 24 x 64 ]
# output : [64 x 24 x 24 x 64 ]
#######################################

res10 = tf.add(h_pool1, h_conv9)
W_conv10 = weight_variable([3, 3, 64, 64])
b_conv10 = bias_variable([64])
r_conv10 = tf.nn.conv2d(res10, W_conv10, strides=[1,1,1,1], padding = 'SAME') + b_conv10
h_conv10 = tf.nn.relu(r_conv10)

#######################################
# HIDDEN LAYER 11 #
# input : [64 x 24 x 24 x 64 ]
# output : [64 x 24 x 24 x 64 ]
#######################################

W_conv11 = weight_variable([3, 3, 64, 64])
b_conv11 = bias_variable([64])
r_conv11 = tf.nn.conv2d(h_conv10, W_conv11, strides=[1,1,1,1], padding = 'SAME') + b_conv11
h_conv11 = tf.nn.relu(r_conv11)


#######################################
# HIDDEN LAYER 12 #
# input : [64 x 24 x 24 x 64 ]
# output : [64 x 24 x 24 x 64 ]
#######################################

W_conv12 = weight_variable([3, 3, 64, 64])
b_conv12 = bias_variable([64])
r_conv12 = tf.nn.conv2d(h_conv11, W_conv12, strides=[1,1,1,1], padding = 'SAME') + b_conv12
h_conv12 = tf.nn.relu(r_conv12)

#######################################
# HIDDEN LAYER 13 # (Residual # 4)
# input : [64 x 24 x 24 x 64 ]
# output : [64 x 24 x 24 x 64 ]
#######################################

res12 = tf.add(h_conv10, h_conv12)
W_conv12 = weight_variable([3, 3, 64, 64])
b_conv12 = bias_variable([64])
r_conv12 = tf.nn.conv2d(res12, W_conv12, strides=[1,1,1,1], padding = 'SAME') + b_conv12
h_conv12 = tf.nn.relu(r_conv12)

#######################################
# fully connected LAYER
# input : [64 x 12 x 12 x 128]
# output : [12 x 12 x 3072]
#######################################

W_fc1 = weight_variable([24 * 24 * 64, 1024])
b_fc1 = bias_variable([1024])
h_conv12_flat = tf.reshape(h_conv12, [-1, 24 * 24 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_conv12_flat, W_fc1) + b_fc1)

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
# Loss : ross entropy
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

f_acc = io.open("test_accuracy_graph.txt", "w", encoding="utf8")
test_acc = list()

global_mean = np.mean(X_train, dtype=np.float64)
if debug_session is True :


    face_mean = np.mean(X_train, axis=0)
    face_mean = np.reshape(face_mean, (48, 48))

    print global_mean
    print face_mean

    imgplot = plt.imshow(face_mean, cmap ="gray")


########################################
# Training
########################################

for i in range(epoch):

    # Each epoch, shuffling all of the data set
    shuffled_indexes = np.arange(len(X_train))
    np.random.shuffle(shuffled_indexes)

    for j in range(len(X_train) // batch_size):

        # Make mini batch
        if j == (len(X_train) // batch_size):
            shuffle_indexes = shuffled_indexes[batch_size * j:len(X_train)]
            X_batch = X_train[shuffle_indexes, :]
            Y_batch = Y_train[shuffle_indexes]
        else:
            shuffle_indexes = shuffled_indexes[batch_size*j:batch_size*(j+1)]
            X_batch = X_train[shuffle_indexes, :]
            Y_batch = Y_train[shuffle_indexes]

        ########################################
        # Add here to augment data
        ########################################

        ########################################
        # Hide-And-Seek
        ########################################
        #for num in range(0, batch_size):
           # X_batch[num,:] = make_hidden_patch(X_batch[num,:], global_mean)

        #test_img = np.reshape(X_batch[0,:] , (48,48))
        #plt.imshow(test_img, cmap = "gray")

        train_step.run(feed_dict= {X: X_batch, y_: Y_batch, keep_prob : 0.5})
        train_accuracy = accuracy.eval(feed_dict = { X : X_batch, y_ : Y_batch, keep_prob : 1.0})

        #print(' training step %d / %d , training acc : %f' % (j*batch_size, len(X_train), train_accuracy))

    test_accuracy = accuracy.eval(feed_dict = { X : X_test, y_ : Y_test, keep_prob : 1.0 })
    test_acc.append([i, test_accuracy])
    print('epoch is %d / %d, test acc : %f' % (i, epoch, test_accuracy))
    np.savetxt('test_accuracy_graph_'+'epoch+'+str(epoch)+'batchSize+'+str(batch_size)+'.txt', np.array(test_acc), fmt='%.2f', delimiter='\t')

########################################
# Save the model
########################################


print ('Training has been done !')
save_path = saver.save(sess, "./training/emotion_model_dnn.ckpt")
print (' Model saved in file : %s' % save_path )

#plt.plot(test_acc)
