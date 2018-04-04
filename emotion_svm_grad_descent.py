import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

def eval_numerical_gradient(f, data, label, x):

    fx = f(data, label, x) # evaluate function value at original point

    grad = np.zeros(x.shape)
    h = 0.00001
    num = 0

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h

        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        fxh = f(data, label, x)
        x[ix] = old_value

        # compute the partial derivative

        grad[ix] = (fxh-fx) / h
        it.iternext()


    return grad

def L_i(X, Y, W, reg):
    ###################################################################
    # W : weight vector [7 x 2304] // x : [2304 x n] // y : [7 x n]   #
    ###################################################################

    loss = 0.0
    dW = np.zeros(W.shape)

    num_train = X.shape[1]
    scores = W.dot(X)
    sequence = np.array(range(num_train))
    correct_class_scores = scores[Y, sequence]

    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[Y, sequence] = 0
    loss = np.sum(margins)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)

    margins = np.where(margins > 0, 1, 0)
    margins[Y, np.arange(0, scores.shape[1])] = -1*np.sum(margins, 0)
    dW = np.dot(margins, X.T)

    dW /= num_train
    dW += reg*W

    return loss

def Emotion_loss_fun(X_batch, Y_batch, W):
    reg = 50
    loss_value = L_i(X_batch, Y_batch, W, reg)
    return loss_value

def predict(X, W):
    y_pred = None
    scores = W.dot(X)
    y_pred = np.argmax(scores, axis=0)

    return y_pred



X_train = np.load('./data/data_set_fer2013.npy')
Y_train = np.load('./data/data_labels_fer2013.npy')
X_test = np.load('./data/test_set_fer2013.npy')
Y_test = np.load('./data/test_labels_fer2013.npy')

image_sample = X_train[:1]
image_sample = image_sample.reshape(48,48)
#plt.imshow(image_sample, cmap = cm.gray)
Xtr_cols = X_train.reshape(48*48, X_train.shape[0])
Xte_cols = X_test.reshape(48*48, X_test.shape[0])


batch_size = 256
step_size = 0.00001
num_iters = 100
W = np.random.rand(7, 2304) * 0.001
num_train = X_train.shape[0]
iterations_per_epoch = max(num_train / batch_size, 1)

loss_history = []
train_acc_history = []
val_acc_history = []

for it in range(num_iters):
    X_batch = None
    Y_batch = None

    shuffle_indexes = np.arange(num_train)
    np.random.shuffle(shuffle_indexes)
    shuffle_indexes = shuffle_indexes[0:batch_size]
    X_batch = X_train[shuffle_indexes,:]
    Y_batch = Y_train[shuffle_indexes]

    X_batch = X_batch.reshape(48*48, batch_size)
    #batch_sample = X_batch.reshape(batch_size, 48*48)
    #batch_sample = batch_sample[:1]
    #batch_sample = batch_sample.reshape(48, 48)
    #plt.imshow(batch_sample , cmap = cm.gray)


    W_grad = eval_numerical_gradient(Emotion_loss_fun, X_batch, Y_batch, W)
    W += - step_size * W_grad
    loss_new = Emotion_loss_fun(X_batch, Y_batch, W)
    print('iteration %d / %d : loss %f' % (it, num_iters, loss_new))

    Wshow = W * 1000 * 255
    template_1 = Wshow[:1]
    template_1 = template_1.reshape(48, 48)

    template_2 = Wshow[1:2]
    template_2 = template_2.reshape(48, 48)

    template_3 = Wshow[2:3]
    template_3 = template_3.reshape(48, 48)

    template_4 = Wshow[3:4]
    template_4 = template_4.reshape(48, 48)



    plt.imshow(template_1, cmap = cm.gray)
    plt.imshow(template_2, cmap = cm.gray)
    plt.imshow(template_3, cmap=cm.gray)
    plt.imshow(template_4, cmap=cm.gray)

    Yte_predict = predict(Xte_cols, W)
    acc = np.mean(Yte_predict == Y_test)
    print('accuracy : %f' % acc)


