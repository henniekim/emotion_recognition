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
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) # L1 Distances
            #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1)) # L2 Distances
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
            print ( 'progress ', i)

        return Ypred

class K_NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, knum):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        k_labels = np.zeros(knum, dtype = self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) # L1 Distances
            #distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1)) # L2 Distances

            sorted_distances = np.sort(distances)
            sorted_index = np.argsort(distances)

            k_distances = sorted_distances[:knum]
            k_index = sorted_index[:knum]

            for j in range(knum):
                k_labels[j] = self.ytr[k_index[j]]

            # vote for the most appearance
            unique_elements, counts_elements = np.unique(k_labels, return_counts = True)
            most_appearance = unique_elements[np.argmax(counts_elements)]
            Ypred[i] = most_appearance
            print ( 'progress ', i)

        return Ypred


Xtr = np.load('./data/data_set_fer2013.npy')
Ytr = np.load('./data/data_labels_fer2013.npy')
Xte = np.load('./data/test_set_fer2013.npy')
Yte = np.load('./data/test_labels_fer2013.npy')

Xtr_rows = Xtr.reshape(Xtr.shape[0], 48*48)
Xte_rows = Xte.reshape(Xte.shape[0], 48*48)

validation_accuracies = []
for k in [1,3,5, 10, 20, 50, 100]:

    knn = K_NearestNeighbor()
    knn.train(Xtr_rows, Ytr)
    Yte_predict = knn.predict(Xte_rows, k)
    acc = np.mean(Yte_predict == Yte)
    print ('accuracy : %f' % (acc,))
    validation_accuracies.append((k, acc))
    
print(' [1,3,5, 10, 20, 50, 100] ')   
print('  ', validation_accuracies)
