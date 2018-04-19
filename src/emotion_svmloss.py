import numpy as np


def L_i(X, Y, W): # W : weight vector [7 x 2304] // x : [2304 x n] // y : [7 x n]
    delta = 1.0
    scores = W.dot(X)
    scores2 = scores[Y, np.arange(scores.shape[1])]
    margins = np.maximum(0, scores - scores[Y, np.arange(scores.shape[1])] + delta )
    margins[Y, np.arange(Y.shape[0])] = 0
    loss = np.sum(margins)

    return loss

Xtr = np.load('./data/data_set_fer2013.npy')
Ytr = np.load('./data/data_labels_fer2013.npy')
Xte = np.load('./data/test_set_fer2013.npy')
Yte = np.load('./data/test_labels_fer2013.npy')

x = Xtr.shape[0]

Xtr_cols = Xtr.reshape(48*48, Xtr.shape[0])
Xte_cols = Xte.reshape(48*48, Xte.shape[0])

bestloss = float("inf")

for num in range(1000):
    W = np.random.randn(7, 2304) * 0.0001
    loss = L_i(Xtr_cols, Ytr, W)
    if loss < bestloss:
        bestloss = loss
        bestW = W
    print (' in attempt %d the loss was %f, best %f' % (num, loss, bestloss))


scores = bestW.dot(Xte_cols)
Yte_predict = np.argmax(scores, axis = 0)
print (' accuracy : %f ' % np.mean(Yte_predict == Yte))
