import numpy as np
import pandas as pd
np.random.seed(0)
from sklearn.metrics import accuracy_score
from scipy.stats import mode

class KNN:
    def read_data(self, normalise):
        data = pd.read_csv("../data/spambase.data.txt", header=None)
        data = data.values
        np.random.shuffle(data)
        if (normalise):
            data = self.__normalise_data(data)
        self.data = data
        train_size = int(data.shape[0] * 0.8)
        X_train = data[:train_size, :-1]
        Y_train = data[:train_size, -1]
        X_test = data[train_size:,:-1]
        Y_test = data[train_size:,-1]

        return X_train,Y_train,X_test, Y_test

    def __normalise_data(self, dataset):
        for i in range(dataset.shape[1] - 1):
            temp = dataset[:, i]
            dataset[:, i] = (temp - np.mean(temp)) / np.std(temp)
        return dataset

    def fit(self, X, Y, distance, k):
        self.X_train = X
        self.Y_train = Y
        self.distance = distance
        self.k = k
        self.most_repeated = mode(self.Y_train)[0][0]

    def euclidean_distance(self, x):
        return np.sqrt(np.sum(np.square(self.X_train - x), axis =1))

    def get_distance(self, x):
        if(self.distance == "euclidean"):
            return self.euclidean_distance(x)

    def get_minimum_distance(self, x):
        distances = self.get_distance(x)
        indices = np.where(distances <= 2.5)[0]
        if(indices.size == 0):
            return self.most_repeated
        within_window =  self.Y_train[indices]

        return mode(within_window)[0][0]

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.get_minimum_distance(X[i]))
        return y_pred

    def run(self, normalise,  distance, k):
        X_train, Y_train, X_test, Y_test = self.read_data(normalise)
        self.fit(X_train, Y_train, distance, k)
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)
        print("training accuracy = ", accuracy_score(Y_train, y_pred_train))
        print("testing accuracy = ", accuracy_score(Y_test, y_pred_test))


knn= KNN()
knn.run(normalise=True, distance="euclidean",k =7)

