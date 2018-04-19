import numpy as np
import matplotlib.pyplot as plt

class emotion(object):

    X_train = np.load('./data/train_set_fer2013_vector.npy')
    Y_train = np.load('./data/train_labels_fer2013_vector.npy')
    X_test = np.load('./data/test_set_fer2013_vector.npy')
    Y_test = np.load('./data/test_labels_fer2013_vector.npy')

    dict (angry=0, disgusted=1, fearful=2, happy=3, sad=4, surprised=5, neutral=6)

    def __init__(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 48 * 48)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 48 * 48)

        self.train_global_mean_face = np.mean(self.X_train, axis = 0)
        self.train_global_mean_face = np.reshape(self.train_global_mean_face, (48, 48))
        self.test_global_mean_face = np.mean(self.X_test, axis=0)
        self.test_global_mean_face = np.reshape(self.test_global_mean_face, (48, 48))
        pass

    def count(self, emotion):
        num = 0
        for i in range(len(self.Y_train)):
            temp = self.Y_train[i, :]
            if temp[emotion] == 1:
                num += 1
        return num

    def meanFace(self, emotion_number):
        num = 0
        emotion_indexes = np.zeros(self.count(emotion_number))
        for i in range(len(self.X_train)):
            temp = self.Y_train[i, :]
            if temp[emotion_number] == 1:
                emotion_indexes[num] = int(i)
                num += 1

        emotion_indexes = np.int_(emotion_indexes)
        X_emotion_array = self.X_train[emotion_indexes, :]
        Y_emotion_array = self.Y_train[emotion_indexes, :]

        emotion_mean = np.mean(X_emotion_array, axis = 0)
        emotion_mean = np.reshape(emotion_mean, (48, 48))

        return emotion_mean

    def sub(self, i,j):
        return np.subtract(self.meanFace(i), self.meanFace(j))

    def DiffPicture(self, save):
        a = np.zeros((7,7))
        for i in range(7):
            for j in range(7):
                grab_subs = self.sub(i,j)
                #plt.imshow(grab_subs, cmap="gray")
                a[i,j]=np.sum(abs(grab_subs))
                if save == True:
                    plt.imsave(str(i)+"_subs_"+str(j)+".png", grab_subs, format = "png")
                    print 'saved successfully'
        print a

# emotion key mapping
angry=0
disgusted=1
fearful=2
happy=3
sad=4
surprised=5
neutral=6

emotion = emotion()
for i in range(7):
    print emotion.count(i)




print 'end'