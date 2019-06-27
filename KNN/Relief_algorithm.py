import numpy as np
import pandas as pd
np.random.seed(42)



class relief():
    def read_data(self, normalise):
        data = pd.read_csv("../data/spambase.data.txt", header=None)
        data = data.values
        np.random.shuffle(data)
        if (normalise):
            data = self.__normalise_data(data)
        self.data = data
        self.X = self.data[:,:-1]
        self.Y = self.data[:,-1]
        self.W = np.zeros(self.X.shape[1])
        train_size = int(data.shape[0] * 0.8)

    def __normalise_data(self, dataset):
        for i in range(dataset.shape[1] - 1):
            temp = dataset[:, i]
            dataset[:, i] = (temp - np.mean(temp)) / np.std(temp)
        return dataset

    def get_distances(self,x):
        return np.sqrt(np.sum(np.square(self.X - x), axis=1))

    def update_W(self, x, y,i):
        distances = self.get_distances(x)
        distances[i] += 1000
        closest_same = self.X[np.argmin(distances[np.where(self.Y == y)])]
        closent_different = self.X[np.argmin(distances[np.where(self.Y != y)])]
        self.W = self.W - np.square(x - closest_same) + np.square(x - closent_different)


    def get_weights(self):
        for i in range(self.X.shape[0]):
            self.update_W(self.X[i], self.Y[i],i)

    def get_top_features(self):
        return self.W.argsort()[-5:][::-1]


    def run(self, normalise):
        self.read_data(normalise=normalise)
        self.get_weights()
        features = self.get_top_features()
        print(features)


relief = relief()
relief.run(False)


