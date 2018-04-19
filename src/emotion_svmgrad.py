import numpy as np

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
        
        num = num + 1
        print (' iteration : %f' % num)

    return grad

def L_i(X, Y, W): # W : weight vector [7 x 2304] // x : [2304 x n] // y : [7 x n]
    delta = 1.0
    scores = W.dot(X)
    margins = np.maximum(0, scores - scores[Y, np.arange(scores.shape[1])] + delta )
    margins[Y, np.arange(Y.shape[0])] = 0
    loss = np.sum(margins)

    return loss

def Emotion_loss_fun(X_train, Y_train, W):
    loss_value = L_i(X_train, Y_train, W)
    return loss_value


X_train = np.load('./data/data_set_fer2013.npy')
Y_train = np.load('./data/data_labels_fer2013.npy')
X_test = np.load('./data/test_set_fer2013.npy')
Y_test = np.load('./data/test_labels_fer2013.npy')

Xtr_cols = X_train.reshape(48*48, X_train.shape[0])
Xte_cols = X_test.reshape(48*48, X_test.shape[0])

W = np.random.rand(7, 2304) * 0.001 # random weight vector
df = eval_numerical_gradient(Emotion_loss_fun, Xtr_cols, Y_train, W) # get the gradient

loss_original = Emotion_loss_fun(Xtr_cols, Y_train, W)
print ('original loss : %f' % loss_original)

for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10 ** step_size_log
    W_new = W - step_size * df
    loss_new = Emotion_loss_fun(Xtr_cols, Y_train, W_new)
    print (" for step size %f new loss : %f " % (step_size, loss_new))


