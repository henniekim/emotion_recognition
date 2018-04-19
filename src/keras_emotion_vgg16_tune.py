from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras

import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#######################################
#             DATA LOAD               #
#######################################

sgd_batch_size = 64
sgd_epoch = 50

X_train = np.load('./data/train_set_fer2013_vector.npy')
Y_train = np.load('./data/train_labels_fer2013_vector.npy')
X_test = np.load('./data/test_set_fer2013_vector.npy')
Y_test = np.load('./data/test_labels_fer2013_vector.npy')

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test= X_test.reshape(X_test.shape[0], 48, 48, 1)

X_train = np.repeat(X_train.reshape(X_train.shape[0], 48, 48, 1), 3, axis=3)
X_test = np.repeat(X_test.reshape(X_test.shape[0], 48, 48, 1), 3, axis=3)

print ('---------------------------------')
print (' Training Data : %d' % len(X_train))
print (' Test Data : %d' % len(X_test))
print ('---------------------------------')


base_model = VGG16(weights='imagenet', include_top= False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation = 'relu')(x)

predictions = Dense(7, activation ='softmax')(x)

model = Model(inputs=base_model.input, outputs = predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, batch_size=64, epochs = 50, validation_data = (X_test, Y_test))


for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in model.layers[:18]:
    layer.trainable = False

for layer in model.layers[18:]:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum = 0.9), loss='categorical_crossentropy', metrics=['accuracy'])

filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose = 0, save_best_only = True, period = 1)
model.save('./model/vgg_16_fine_tune_'+'batchsize_'+str(sgd_batch_size)+'epoch_'+str(sgd_epoch))


model.fit(x=X_train, y=Y_train, batch_size= sgd_batch_size, epochs = sgd_epoch, validation_data = (X_test, Y_test))

