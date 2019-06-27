import numpy as np
import pandas as pd
np.random.seed(42)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import mode

class KNN:
    def read_data(self):
        X = np.loadtxt("../data/Digits_dataset_20_percent_features.txt")
        Y = np.loadtxt("../data/Digits_dataset_20_percent_labels.txt")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        data = np.c_[X, Y]
        np.random.shuffle(data)
        train_size = int(data.shape[0] * 0.8)
        X_train = data[:train_size, :-1]
        X_test = data[train_size:, :-1]
        Y_train = data[:train_size, -1]
        Y_test = data[train_size:, -1]
        return X_train, Y_train, X_test, Y_test

    def fit(self,X_train,Y_train, distance, k ):
        self.X_train, self.Y_train, self.distance, self.k = X_train,Y_train, distance, k
        self.norm_X = np.linalg.norm(self.X_train, axis = 1)
        self.most_repeated = mode(self.Y_train)[0][0]

    def euclidean_distance(self, x):
        return np.sqrt(np.sum(np.square(self.X_train - x), axis =1))

    def cosine_distance(self, x):
        point_norm = np.linalg.norm(x)
        return 1 - np.divide(np.sum(np.multiply(self.X_train, x), axis = 1),  (np.multiply(self.norm_X, point_norm)))

    def gaussian(self, x):
        return np.exp(np.linalg.norm(self.X_train - x, ord= 2, axis = 1)) / (2 * ((0.1 ** 2)))

    def poly(self,x):
        return 1 - np.square(np.sum(np.multiply(self.X_train, x), axis = 1) + 1)

    def get_distance(self, x):
        if(self.distance == "euclidean"):
            return self.euclidean_distance(x)
        elif(self.distance == "cosine"):
            return self.cosine_distance(x)
        elif(self.distance == "gaussian"):
            return self.gaussian(x)
        elif(self.distance == "poly"):
            return self.poly(x)


    def get_minimum_distance(self, x):
        distances = self.get_distance(x)
        indices = np.where(distances <= 0.3)[0]
        if (indices.size == 0):
            return self.Y_train[np.argmin(distances)]
        within_window = self.Y_train[indices]

        return mode(within_window)[0][0]

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.get_minimum_distance(X[i]))
            if(i % 1000  == 0):
                print(i)
        return y_pred


    def run(self, distance, k):
        X_train, Y_train, X_test, Y_test= self.read_data()
        self.fit(X_train,Y_train, distance, k)
        print("done reading")
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)
        print("training accuracy = ", accuracy_score(Y_train, y_pred_train))
        print("testing accuracy = ", accuracy_score(Y_test, y_pred_test))



knn= KNN()
knn.run(distance="cosine",k =7)