import pandas as pd
import numpy as np
import sklearn
np.random.seed(0)
from sklearn.svm import SVC

class SVM_Built_In:
    def read_data(self, normalise):
        data = pd.read_csv("../data/spambase.data.txt", header=None)
        data = data.values
        np.random.shuffle(data)
        if(normalise):
            data = self.__normalise_data(data)
        self.data = data

    def __normalise_data(self, dataset):
        for i in range(dataset.shape[1] - 1):
            temp = dataset[:, i]
            dataset[:, i] = (temp - np.mean(temp)) / np.std(temp)
        return dataset

    def SVM(self, kernel):
        train_size = int(self.data.shape[0]  * 0.8)
        X_train = self.data[:train_size,:-1]
        Y_train = self.data[:train_size,-1]
        X_test = self.data[train_size:, :-1]
        Y_test = self.data[train_size:,-1]
        classifier = SVC(C = 5, kernel=kernel, gamma='auto', degree=2)
        classifier.fit(X_train, Y_train)
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
        print("Training_accuracy for kernel ", kernel, " is = ",  sklearn.metrics.accuracy_score(Y_train, y_pred_train))
        print("Testing_accuracy for kernel ", kernel, " is = ", sklearn.metrics.accuracy_score(Y_test, y_pred_test))

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svm = SVM_Built_In()
    svm.read_data(normalise = True)
    svm.SVM(kernels[i])


