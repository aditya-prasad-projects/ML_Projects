import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

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

def convert_one_zero(X_train, Y_train):
     mu = np.sum(X_train, axis = 0) / X_train.shape[0]
     X_train = np.where(X_train <= mu, 0, X_train)
     X_train = np.where(X_train > mu, 1, X_train)
     return X_train, Y_train

def fit_bernoulli_distribution(X_train,Y_train):
    X,Y = convert_one_zero(X_train, Y_train)
    X_Y1 = X[np.where(Y==1)]
    X_Y0 = X[np.where(Y==0)]
    Y1 = np.count_nonzero(Y)
    Y0 = np.count_nonzero(Y == 0)
    count_X1_Y1 = np.count_nonzero(X_Y1, axis=0)
    count_X0_Y1 = np.count_nonzero(X_Y1 == 0, axis=0)
    count_X1_Y0 = np.count_nonzero(X_Y0, axis = 0)
    count_X0_Y0 = np.count_nonzero(X_Y0 == 0, axis = 0)
    #probability_X1_Y1 = (count_X1_Y1) / (Y1)
    #probability_X0_Y1 = (count_X0_Y1) / (Y1)
    #probability_X1_Y0 = (count_X1_Y0) / (Y0)
    #probability_X0_Y0 = (count_X0_Y0) / (Y0)
    probability_X1_Y1 = (count_X1_Y1 + 1) / (Y1 + X.shape[1])
    probability_X0_Y1 = (count_X0_Y1 + 1) / (Y1 + X.shape[1])
    probability_X1_Y0 = (count_X1_Y0 + 1) / (Y0 + X.shape[1])
    probability_X0_Y0 = (count_X0_Y0 + 1) / (Y0 + X.shape[1])
    return probability_X1_Y1, probability_X0_Y1, probability_X1_Y0, probability_X0_Y0

def plot_ROC_curve(probability_Y0, probability_Y1, Y):
    log_probabilities = np.log2(probability_Y1 / probability_Y0)
    sorted_indices = np.argsort(log_probabilities)
    log_probabilities.sort()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y[sorted_indices], log_probabilities)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("AUC = ", roc_auc)



def get_prediction(X_train,Y_train,probability_X1_Y1, probability_X0_Y1, probability_X1_Y0, probability_X0_Y0, k):
    X, Y = convert_one_zero(X_train, Y_train)
    Y1 = np.count_nonzero(Y) / Y.shape[0]
    Y0 = np.count_nonzero(Y==0) / Y.shape[0]
    y_pred = []
    probability_Y1 = []
    probability_Y0 = []
    for i in range(X_train.shape[0]):
        prob_y1 = 0
        prob_y0 = 0
        for j in range(X_train.shape[1]):
            if(X[i][j] == 0):
                prob_y1 = prob_y1 + np.log(probability_X0_Y1[j])
                prob_y0 = prob_y0 + np.log(probability_X0_Y0[j])
            if(X[i][j] == 1):
                prob_y1 = prob_y1 + np.log(probability_X1_Y1[j])
                prob_y0 = prob_y0 + np.log(probability_X1_Y0[j])
        prob_Y1_X = ((prob_y1) + np.log(Y1)) / ((prob_y1 * Y1) + (prob_y0 * Y0))
        prob_Y0_X = ((prob_y0) + np.log(Y0)) / ((prob_y1 * Y1) + (prob_y0 * Y0))
        probability_Y0.append(prob_Y1_X)
        probability_Y1.append(prob_Y0_X)
        if(prob_Y0_X > prob_Y1_X):
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    accuracy = sklearn.metrics.accuracy_score(Y, y_pred)
    print("accuracy = ", accuracy)
    if(k == 0):
        plot_ROC_curve(np.array(probability_Y0), np.array(probability_Y1), Y_train)
    return accuracy

def k_split(data, k_split):
    k = [i for i in range(k_split)]
    n = np.array_split(data, k_split)
    training_accuracy = 0
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
        probability_X1_Y1, probability_X0_Y1, probability_X1_Y0, probability_X0_Y0 = fit_bernoulli_distribution(X_train,Y_train)
        training_accuracy = training_accuracy + get_prediction(X_train,Y_train, probability_X1_Y1, probability_X0_Y1, probability_X1_Y0, probability_X0_Y0, i)
        test_accuracy = test_accuracy + get_prediction(X_test, Y_test, probability_X1_Y1, probability_X0_Y1, probability_X1_Y0, probability_X0_Y0, i)
    print("training accuracy = ", training_accuracy / k_split)
    print("testing accuracy = ", test_accuracy / k_split)

def run():
    data  = read_data("../data/spambase.data.txt")
    #normalised_data = normalise_data(data)
    k_split(data, 10)


run()