import pandas as pd
import numpy as np
import sklearn.metrics

np.random.seed(10)


def read_data(file_name):
    entire_dataset = pd.read_csv(file_name, header=None)
    dataset = entire_dataset.values
    np.random.shuffle(dataset)
    return dataset

def normalise_data(data):
    r,c = data.shape
    for i in range(c - 1):
        temp = data[:, i]
        data[:, i] = (temp - np.mean(temp)) / np.std(temp)
    return data

def fit_gaussian_curves(X_train, Y_train):
    X_train_zero = X_train[np.where(Y_train == 0)]
    X_train_one = X_train[np.where(Y_train == 1)]
    mu0 = np.sum(X_train_zero, axis = 0) / (Y_train.shape[0] - np.sum(Y_train))
    mu1 = np.sum(X_train_one, axis = 0) / np.sum(Y_train)
    variance0 = X_train_zero - mu0
    sigma0 = (np.dot(variance0.T, variance0)) / (Y_train.shape[0] - np.sum(Y_train))
    variance1 = X_train_one - mu1
    sigma1 = (np.dot(variance1.T, variance1)) / np.sum(Y_train)
    np.fill_diagonal(sigma0, sigma0.diagonal() + 0.3)
    np.fill_diagonal(sigma1, sigma1.diagonal() + 0.3)
    return mu0, mu1, sigma0,sigma1

def get_predictions(X, Y, Y_train, mu0, mu1, sigma0, sigma1):
    denominatorpi = np.power((2 * np.pi), (X.shape[1] / 2))
    denominator0 = np.log2(1  / (np.sqrt(np.linalg.det(sigma0)) * denominatorpi))
    denominator1 = np.log2(1  / (np.sqrt(np.linalg.det(sigma1)) * denominatorpi))
    sigmaInverse0 = np.linalg.pinv(sigma0)
    sigmaInverse1 = np.linalg.pinv(sigma1)
    py0 = (Y_train.shape[0] - np.sum(Y_train)) / Y_train.shape[0]
    py1 = (np.sum(Y_train)) / Y_train.shape[0]
    y_pred = []
    for i in range(X.shape[0]):
        temp = X[i]
        variance0 = (temp - mu0)
        pxy0 = denominator0 * np.dot(np.dot(variance0.T, sigmaInverse0), variance0)
        pyx0 = pxy0 * py0
        variance1 = temp - mu1
        pxy1 = denominator1 * np.dot(np.dot(variance1.T, sigmaInverse1), variance1)
        pyx1 = pxy1 * py1
        if(pyx0 > pyx1):
            y_pred.append(0)
        else:
            y_pred.append(1)
    y_pred = np.array(y_pred)
    accuracy = sklearn.metrics.accuracy_score(Y, y_pred)
    print("accuracy = ", accuracy)
    return accuracy

def k_split(data, k_split):
    k = [i for i in range(k_split)]
    n = np.array_split(data, k_split)
    training_error = 0
    training_accuracy = 0
    test_error = 0
    test_accuracy = 0
    for i in range(k_split):
        data = n[i]
        X_test = data[:, :-1]
        Y_test = data[:, -1]
        counter = 0
        for j in range(k_split):
            if (j != i):
                if (counter == 0):
                    data = n[j]
                    counter += 1
                else:
                    data = np.concatenate((data, n[j]), axis=0)
        X_train = data[:, :-1]
        Y_train = data[:, -1]
        mu0, mu1, sigma0, sigma1 = fit_gaussian_curves(X_train, Y_train)
        training_accuracy = training_accuracy + get_predictions(X_train, Y_train, Y_train, mu0, mu1, sigma0, sigma1)
        test_accuracy = test_accuracy + get_predictions(X_test, Y_test, Y_train, mu0, mu1,sigma0,sigma1)
    print("training_accuracy  = ", training_accuracy / k_split)
    print("test_accuracy = ", test_accuracy / k_split)

def run():
    data  = read_data("../data/spambase.data.txt")
    normalised_data = normalise_data(data)
    k_split(normalised_data, 10)


run()


