import pandas as pd
import numpy as np
import sklearn
np.random.seed(42)
from SVM.SMO import SMO
from sklearn.datasets import make_blobs
import pylab
import matplotlib.pyplot as plt




class SVM:
    def __read_data(self, normalise):
        data = pd.read_csv("../data/spambase.data.txt", header=None)
        data = data.values
        np.random.shuffle(data)
        if(normalise):
            data = self.__normalise_data(data)
        self.data = data

        return data

    def __normalise_data(self, dataset):
        for i in range(dataset.shape[1] - 1):
            temp = dataset[:, i]
            dataset[:, i] = (temp - np.mean(temp)) / np.std(temp)
        return dataset

    def __plot(self, X, Y):
        plt.scatter(X,Y)
        plt.show()


    def _generate_data(self):
        X, Y = make_blobs()
        self.__plot(X, Y)
        return np.c_[X, Y]


    def run(self,normalise, k_split):
        epochs = [71,201,51,201,201,201,31,1,11,91]
        with open("SVM.txt", "w") as f:
            data = self.__read_data(normalise)
            data[:,-1] = np.where(data[:,-1] <= 0, -1, 1)
            n = np.array_split(data, k_split)
            training_accuracy = 0
            testing__accuracy = 0
            for i in range(k_split):
                print("K_split = ", i)
                f.write("\nK_split = " + str(i))
                data_temp = n[i]
                X_test = data_temp[:, :-1]
                Y_test = data_temp[:, -1]
                counter = 0
                for j in range(k_split):
                    if (j != i):
                        if (counter == 0):
                            data_temp = n[j]
                            counter += 1
                        else:
                            data_temp = np.concatenate((data_temp, n[j]), axis=0)
                X_train = data_temp[:, :-1]
                Y_train = data_temp[:, -1]
                svm_smo = SMO(C =0.01, tol = 0.01, max_passes = 100, epochs = 100, f = f)
                svm_smo.fit(X_train, Y_train)
                y_pred_train = svm_smo.predict(X_train)
                y_pred_test = svm_smo.predict(X_test)
                train_accuracy_temp = sklearn.metrics.accuracy_score(Y_train, y_pred_train)
                test_accuracy_temp = sklearn.metrics.accuracy_score(Y_test, y_pred_test)
                training_accuracy += train_accuracy_temp
                testing__accuracy += test_accuracy_temp
                f.write("\nTraining accuracy = " + str(train_accuracy_temp))
                f.write("\nTesting accuracy = " + str(test_accuracy_temp))
                print("Training accuracy = ", train_accuracy_temp)
                print("Testing accuracy = ", test_accuracy_temp)
            training_accuracy = training_accuracy / k_split
            testing__accuracy = testing__accuracy / k_split
            f.write("\nAverage training accuracy = " + str(training_accuracy))
            f.write("\nAverage training accuracy = " + str(testing__accuracy))
            print("Average training accuracy = ", training_accuracy)
            print("Average testing accuracy = ", testing__accuracy)


svm = SVM()
svm.run(normalise=True, k_split=10)
