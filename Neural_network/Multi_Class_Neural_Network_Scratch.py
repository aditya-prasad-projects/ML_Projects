import pandas as pd
import numpy as np

np.random.seed(2)

def normalise_data(data):
    r,c = data.shape
    for i in range(c):
        temp = data[:, i]
        data[:, i] = (temp - np.mean(temp)) / np.std(temp)

def read_data(train_file, test_file):
    data_train = pd.read_csv(train_file, header = None)
    X_train = data_train.iloc[:, 1:].values
    Y_train = data_train.iloc[:,0].values
    data_test = pd.read_csv(test_file, header=None)
    X_test = data_test.iloc[:, 1:].values
    Y_test = data_test.iloc[:,0].values
    return X_train,Y_train, X_test, Y_test

def get_one_hot_matrix(Y, total_classes):
    Y_one_hot = np.zeros((Y.shape[0], total_classes))
    training_examples = Y.shape[0]
    for i in range(training_examples):
        Y_one_hot[i,Y[i] - 1] = 1.0
    return Y_one_hot

def activation_function(Z, i):
    return 1 / (1 + np.exp(-Z))

def initialize_parameters(nodes, layers):
    parameters = {}
    for i in range(1, layers + 1):
        parameters["W" + str(i)] = np.random.normal(size=(nodes[i], nodes[i-1]))
        parameters["b" + str(i)] = np.zeros((nodes[i], 1))
    return parameters

def multi_variable_classifier(parameters, activation, layers):
    for i in range(1,layers+1):
        Z = np.matmul(parameters["W" + str(i)], activation["A" + str(i-1)]) + parameters["b" + str(i)]
        activation["A" + str(i)] = activation_function(Z, i)
    return parameters,activation

def backward_propagation(activation, X, Y, parameters, learning_rate):
    m = activation["A2"].shape[1]
    #dZ2 = activation["A2"] - Y
    dZ2 = np.multiply((activation["A2"] - Y), np.multiply(activation["A2"], (1 - activation["A2"])))
    dW2 = (1 / m) * np.matmul(dZ2, activation["A1"].T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.matmul(parameters["W2"].T, dZ2), np.multiply(activation["A1"], (1 - activation["A1"])))
    dW1 = (1 / m) * np.matmul(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    parameters["W1"] = parameters["W1"] - (learning_rate * dW1)
    parameters["b1"] = parameters["b1"] - (learning_rate * db1)
    parameters["W2"] = parameters["W2"] - (learning_rate * dW2)
    parameters["b2"] = parameters["b2"] - (learning_rate * db2)
    return parameters

def loss(y_pred, Y):
    m = y_pred.shape[1]
    #cost = (-1 / m) * np.sum(np.multiply(Y, np.log(y_pred)) + np.multiply((1 - Y), np.log(1 - y_pred)))
    cost = (1/m) * np.sum(np.square(Y-y_pred))

    return cost

def get_accuracy(y_pred,Y):
    counter = 0
    for i in range(len(y_pred)):
        count = 0
        for j in range(3):
            if(y_pred[i][j] == Y[i][j]):
                count = count + 1
        if(count == 3):
            counter +=1
    return counter / len(y_pred)

def get_classifier_matrix(out):
    y_pred = np.zeros(out.shape)
    for i in range(out.shape[0]):
        max_index = np.argmax(out[i])
        y_pred[i][max_index] = 1
    return y_pred.T

def run(X_train, Y_train ,X_test, Y_test, nodes, layers, learning_rate, epochs):
    parameters = initialize_parameters(nodes,layers)
    for i in range(epochs):
        activation = {"A0": X_train}
        parameters,activation = multi_variable_classifier(parameters, activation,layers)
        cost = loss(activation["A" + str(layers)], Y_train)
        parameters = backward_propagation(activation, X_train, Y_train, parameters, learning_rate)
        if (i % 10000 == 0):
            print("i = ", i)
            print("cost = ", cost)
            out = activation["A" + str(layers)]
            y_pred = get_classifier_matrix(out.T)
            train_accuracy = get_accuracy(y_pred.T, Y_train.T)
            print("train_accuracy = ", train_accuracy)
            activation["A0"] = X_test
            _,activation = multi_variable_classifier(parameters,activation,layers)
            test_out = activation["A" + str(layers)]
            y_test_pred = get_classifier_matrix(test_out.T)
            test_accuracy = get_accuracy(y_test_pred.T, Y_test.T)
            print("test_accuracy = ", test_accuracy)

def get_nodes(X, Y):
    features = X.shape[0]
    output = Y.shape[0]
    nodes = [features, 35, output]
    return nodes


def call_everthing():
    X_train, Y_train, X_test, Y_test = read_data("../data/train_wine.csv", "../data/test_wine.csv")
    Y_train = get_one_hot_matrix(Y_train, 3)
    Y_test = get_one_hot_matrix(Y_test, 3)
    X_train = X_train
    X_test = X_test
    X_train, Y_train, X_test, Y_test = X_train.T, Y_train.T, X_test.T, Y_test.T

    nodes = get_nodes(X_train, Y_train)
    run(X_train, Y_train ,X_test, Y_test, nodes, 2, 1e-3, 300000)

call_everthing()
