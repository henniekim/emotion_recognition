import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train( self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) # L1 Distances
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1)) # L2 Distances
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
            print ( 'progress ', i)

        return Ypred

Xtr = np.load('./data/data_set_fer2013.npy')
Ytr = np.load('./data/data_labels_fer2013.npy')
Xte = np.load('./data/test_set_fer2013.npy')
Yte = np.load('./data/test_labels_fer2013.npy')

Xtr_rows = Xtr.reshape(Xtr.shape[0], 48*48)
Xte_rows = Xte.reshape(Xte.shape[0], 48*48)

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)

print ('accuracy : %f' % (np.mean(Yte_predict == Yte)))