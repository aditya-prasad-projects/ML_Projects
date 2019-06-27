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

def get_bin_limits(X_train, Y_train):
    X_mean = np.sum(X_train, axis=0) / X_train.shape[0]
    X_Y1 = X_train[np.where(Y_train == 1)]
    X_Y0 = X_train[np.where(Y_train == 0)]
    X_mean_Y1 = np.sum(X_Y1, axis=0) / X_Y1.shape[0]
    X_mean_Y0 = np.sum(X_Y0, axis=0) / X_Y0.shape[0]
    X_lower_mean = []
    X_upper_mean = []
    for i in range(X_train.shape[1]):
        if (X_mean_Y0[i] < X_mean_Y1[i]):
            X_lower_mean.append(X_mean_Y0[i])
            X_upper_mean.append(X_mean_Y1[i])
        else:
            X_lower_mean.append(X_mean_Y1[i])
            X_upper_mean.append(X_mean_Y0[i])
    X_lower_mean = np.array(X_lower_mean)
    X_upper_mean = np.array(X_upper_mean)
    return X_Y1, X_Y0, X_mean, X_lower_mean, X_upper_mean

def get_counts(X_train, Y_train):
    Y1 = np.count_nonzero(Y_train)
    Y0 = np.count_nonzero(Y_train == 0)
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_Y1, X_Y0, X_mean, X_lower_mean, X_upper_mean = get_bin_limits(X_train, Y_train)
    X1_Y1 = np.count_nonzero(X_Y1 <= X_lower_mean, axis = 0)
    X2_Y1 = np.count_nonzero(np.logical_and(X_Y1 > X_lower_mean, X_Y1 <= X_mean), axis = 0)
    X3_Y1 = np.count_nonzero(np.logical_and(X_Y1 > X_mean, X_Y1 <= X_upper_mean), axis = 0)
    X4_Y1 = np.count_nonzero(X_Y1 > X_upper_mean, axis = 0)
    X1_Y0 = np.count_nonzero(X_Y0 <= X_lower_mean, axis = 0)
    X2_Y0 = np.count_nonzero(np.logical_and(X_Y0 > X_lower_mean, X_Y0 <= X_mean), axis = 0)
    X3_Y0 = np.count_nonzero(np.logical_and(X_Y0 > X_mean, X_Y0 <= X_upper_mean), axis = 0)
    X4_Y0 = np.count_nonzero(X_Y0 > X_upper_mean, axis = 0)
    return X1_Y1,X1_Y0,X2_Y1,X2_Y0,X3_Y1,X3_Y0,X4_Y1,X4_Y0

def fit_histogram(X_train,Y_train):
    probabilities_dictionary = {}
    Y1 = np.count_nonzero(Y_train)
    Y0 = np.count_nonzero(Y_train == 0)
    count_X1_Y1, count_X1_Y0, count_X2_Y1, count_X2_Y0, count_X3_Y1, count_X3_Y0, count_X4_Y1, count_X4_Y0 = get_counts(X_train, Y_train)
    probabilities_dictionary["probability_X1_Y1"] = (count_X1_Y1 + 1) / (Y1 + X_train.shape[1])
    probabilities_dictionary["probability_X2_Y1"] = (count_X2_Y1 + 1) / (Y1 + X_train.shape[1])
    probabilities_dictionary["probability_X3_Y1"] = (count_X3_Y1 + 1) / (Y1 + X_train.shape[1])
    probabilities_dictionary["probability_X4_Y1"] = (count_X4_Y1 + 1) / (Y1 + X_train.shape[1])
    probabilities_dictionary["probability_X1_Y0"] = (count_X1_Y0 + 1) / (Y0 + X_train.shape[1])
    probabilities_dictionary["probability_X2_Y0"] = (count_X2_Y0 + 1) / (Y0 + X_train.shape[1])
    probabilities_dictionary["probability_X3_Y0"] = (count_X3_Y0 + 1) / (Y0 + X_train.shape[1])
    probabilities_dictionary["probability_X4_Y0"] = (count_X4_Y0 + 1) / (Y0 + X_train.shape[1])
    return probabilities_dictionary

def get_prediction(X_train,Y_train, probabilities_dictionary, k):
    Y1 = np.count_nonzero(Y_train) / Y_train.shape[0]
    Y0 = np.count_nonzero(Y_train == 0) / Y_train.shape[0]
    X_Y1, X_Y0, X_mean, X_lower_mean, X_upper_mean = get_bin_limits(X_train, Y_train)
    y_pred = []
    probability_Y1 = []
    probability_Y0 = []
    for i in range(X_train.shape[0]):
        prob_y1 = 0
        prob_y0 = 0
        for j in range(X_train.shape[1]):
            if(X_train[i][j] <= X_lower_mean[j]):
                prob_y1 = prob_y1 + np.log2(probabilities_dictionary["probability_X1_Y1"][j])
                prob_y0 = prob_y0 + np.log2(probabilities_dictionary["probability_X1_Y0"][j])
            elif(X_train[i][j] > X_lower_mean[j] and X_train[i][j] <= X_mean[j]):
                prob_y1 = prob_y1 + np.log2(probabilities_dictionary["probability_X2_Y1"][j])
                prob_y0 = prob_y0 + np.log2(probabilities_dictionary["probability_X2_Y0"][j])
            elif(X_train[i][j] > X_mean[j] and X_train[i][j] <= X_upper_mean[j]):
                prob_y1 = prob_y1 + np.log2(probabilities_dictionary["probability_X3_Y1"][j])
                prob_y0 = prob_y0 + np.log2(probabilities_dictionary["probability_X3_Y0"][j])
            elif(X_train[i][j] > X_upper_mean[j]):
                prob_y1 = prob_y1 + np.log2(probabilities_dictionary["probability_X4_Y1"][j])
                prob_y0 = prob_y0 + np.log2(probabilities_dictionary["probability_X4_Y0"][j])
        prob_Y1_X = ((prob_y1) + np.log2(Y1)) / ((prob_y1 * Y1) + (prob_y0 * Y0))
        prob_Y0_X = ((prob_y0) + np.log2(Y0)) / ((prob_y1 * Y1) + (prob_y0 * Y0))
        probability_Y0.append(prob_Y1_X)
        probability_Y1.append(prob_Y0_X)
        if (prob_Y0_X > prob_Y1_X):
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    accuracy = sklearn.metrics.accuracy_score(Y_train, y_pred)
    if (k == 0):
        plot_ROC_curve(np.array(probability_Y0), np.array(probability_Y1), Y_train)
    print("accuracy = ", accuracy)
    return accuracy

def plot_ROC_curve(probability_Y0, probability_Y1, Y):
    log_probabilities = np.log2(probability_Y1 / probability_Y0)
    #sorted_indices = np.argsort(log_probabilities)
    #log_probabilities.sort()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, log_probabilities)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("AUC = ", roc_auc)






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
        probabilities_dictionary = fit_histogram(X_train,Y_train)
        training_accuracy = training_accuracy + get_prediction(X_train,Y_train, probabilities_dictionary, i)
        test_accuracy = test_accuracy + get_prediction(X_test, Y_test, probabilities_dictionary, i)
    print("training accuracy = ", training_accuracy / k_split)
    print("testing accuracy = ", test_accuracy / k_split)

def run():
    data  = read_data("../data/spambase.data.txt")
    #normalised_data = normalise_data(data)
    k_split(data, 10)


run()