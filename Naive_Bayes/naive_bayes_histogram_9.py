import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

np.random.seed(0)

def read_data(file_name):
    entire_dataset = pd.read_csv(file_name, header=None)
    dataset = entire_dataset.values
    np.random.shuffle(dataset)
    return dataset

def get_bin_limits(X_train, Y_train):
    bin_limits = []
    for i in range(X_train.shape[1]):
        bin_limits.append(np.linspace(np.amin(X_train[:, i], axis=0), np.amax(X_train[:, i], axis=0), 10))
    return np.array(bin_limits).T

def get_counts(X_train, Y_train):
    X_Y1 = X_train[np.where(Y_train == 1)]
    X_Y0 = X_train[np.where(Y_train == 0)]
    bin_limits = get_bin_limits(X_train, Y_train)
    count_dictionary = {}
    count_dictionary["X0_Y1"] = np.count_nonzero(X_Y1 <= bin_limits[1], axis = 0)
    count_dictionary["X0_Y0"] = np.count_nonzero(X_Y0 <= bin_limits[1], axis = 0)
    for i in range(1,9):
        count_dictionary["X" + str(i) + "_Y1"] = np.count_nonzero(np.logical_and(X_Y1 > bin_limits[i], X_Y1 <= bin_limits[i + 1]), axis = 0)
        count_dictionary["X" + str(i) + "_Y0"] = np.count_nonzero(np.logical_and(X_Y0 > bin_limits[i], X_Y0 <= bin_limits[i + 1]), axis = 0)
    count_dictionary["X8_Y1"] = np.count_nonzero(X_Y1 > bin_limits[8], axis = 0)
    count_dictionary["X8_Y0"] = np.count_nonzero(X_Y0 > bin_limits[8], axis = 0)
    return count_dictionary,bin_limits

def fit_histogram(X_train,Y_train):
    probabilities_dictionary = {}
    Y1 = np.count_nonzero(Y_train)
    Y0 = np.count_nonzero(Y_train == 0)
    count_dictionary, bin_limits = get_counts(X_train, Y_train)
    for i in range(9):
        probabilities_dictionary["X" + str(i) + "_Y1"] = (count_dictionary["X" + str(i) + "_Y1"] + 1) / (Y1 + X_train.shape[1])
        probabilities_dictionary["X" + str(i) + "_Y0"] = (count_dictionary["X" + str(i) + "_Y0"] + 1) / (Y0 + X_train.shape[1])
    return probabilities_dictionary, bin_limits

def get_prediction(X_train,Y_train, probabilities_dictionary, bin_limits, l):
    Y1 = np.count_nonzero(Y_train) / Y_train.shape[0]
    Y0 = np.count_nonzero(Y_train == 0) / Y_train.shape[0]
    y_pred = []
    probability_Y1 = []
    probability_Y0 = []
    for i in range(X_train.shape[0]):
        prob_y1 = 0
        prob_y0 = 0
        mini = 0
        for j in range(X_train.shape[1]):
            for k in range(1, bin_limits.shape[1]):
                if(X_train[i][j] <= bin_limits[j][k]):
                    mini = k - 1
                    break
            prob_y1 = (prob_y1 + np.log2(probabilities_dictionary["X" + str(mini) + "_Y1"][j]))
            prob_y0 = (prob_y0 + np.log2(probabilities_dictionary["X" + str(mini) + "_Y0"][j]))
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
    print("accuracy = ", accuracy)
    if (l == 0):
        plot_ROC_curve(np.array(probability_Y0), np.array(probability_Y1), Y_train)
    return accuracy

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
        probabilities_dictionary, bin_limits= fit_histogram(X_train,Y_train)
        training_accuracy = training_accuracy + get_prediction(X_train,Y_train, probabilities_dictionary, bin_limits.T, i)
        test_accuracy = test_accuracy + get_prediction(X_test, Y_test, probabilities_dictionary, bin_limits.T, i)
    print("training accuracy = ", training_accuracy / k_split)
    print("testing accuracy = ", test_accuracy / k_split)
    return training_accuracy, test_accuracy

def run():
    data  = read_data("../data/spambase.data.txt")
    #normalised_data = normalise_data(data)
    training_accuracy, testing_accuracy = k_split(data, 10)

run()