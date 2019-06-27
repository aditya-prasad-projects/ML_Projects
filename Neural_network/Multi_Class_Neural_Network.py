import tensorflow as tf
import numpy as np
import pandas as pd

def get_one_hot_matrix(Y, total_classes):
    Y_one_hot = np.zeros((Y.shape[0], total_classes))
    training_examples = Y.shape[0]
    for i in range(training_examples):
        Y_one_hot[i,Y[i] - 1] = 1.0
    return Y_one_hot

def activation_function(Z, i, layers):
    return tf.nn.sigmoid(Z)



def initialize_parameters(nodes, layers,  X, Y):
    parameters = {}
    for i in range(1, layers + 1):
        parameters["W" + str(i)] = tf.get_variable("W" + str(i), (nodes[i], nodes[i-1]), initializer = tf.random_normal_initializer(seed = 42))
        parameters["b" + str(i)] = tf.get_variable("b" + str(i), (nodes[i], 1), initializer = tf.zeros_initializer)
    return parameters



def multi_variable_classifier(parameters, activation, layers):
    for i in range(1,layers+1):
        Z = tf.matmul(parameters["W" + str(i)], activation["A" + str(i-1)]) + parameters["b" + str(i)]
        activation["A" + str(i)] = activation_function(Z, i,layers)
    return parameters,activation

def loss(y_pred, Y):
    cost = tf.losses.mean_squared_error(Y,y_pred)
    return cost

def run(X_train, Y_train, X_test, Y_test, nodes,layers,learning_rate,epochs):
    X = tf.placeholder(tf.float32, [13, None])
    Y = tf.placeholder(tf.float32, [3,None])
    parameters = initialize_parameters(nodes, layers, X, Y)
    activation = {"A0":X}
    parameters, activation = multi_variable_classifier(parameters, activation, layers)
    cost = loss(activation["A" + str(layers)], Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {X: X_train, Y:Y_train}
        for i in range(epochs):
            _, loss_val = sess.run([optimizer,cost], feed_dict)
            if (i % 5000 == 0):
                print("cost = ", loss_val)
        out = sess.run(activation["A" + str(layers)], feed_dict)
        y_pred = get_classifier_matrix(out.T)
        train_accuracy = get_accuracy(y_pred.T,Y_train.T)
        print("train_accuracy = ", train_accuracy)
        test_out = sess.run(activation["A" + str(layers)], {X: X_test, Y: Y_test})
        y_test_pred = get_classifier_matrix(test_out.T)
        test_accuracy = get_accuracy(y_test_pred.T, Y_test.T)
        print("test_accuracy = ", test_accuracy)

def get_accuracy(y_pred,Y_train):
    counter = 0
    for i in range(len(y_pred)):
        count = 0
        for j in range(3):
            if(y_pred[i][j] == Y_train[i][j]):
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



def read_data(train_file, test_file):
    data_train = pd.read_csv(train_file, header = None)
    X_train = data_train.iloc[:, 1:].values
    Y_train = data_train.iloc[:,0].values
    data_test = pd.read_csv(test_file, header=None)
    X_test = data_test.iloc[:, 1:].values
    Y_test = data_test.iloc[:,0].values
    return X_train,Y_train, X_test, Y_test

def get_nodes(X, Y):
    features = X.shape[0]
    output = Y.shape[0]
    nodes = [features, 25, output]
    return nodes

def call_everthing():
    X_train, Y_train, X_test, Y_test = read_data("../data/train_wine.csv", "../data/test_wine.csv")
    Y_train = get_one_hot_matrix(Y_train, 3)
    Y_test = get_one_hot_matrix(Y_test, 3)
    X_train, Y_train, X_test, Y_test = X_train.T, Y_train.T, X_test.T, Y_test.T

    nodes = get_nodes(X_train, Y_train)
    run(X_train, Y_train ,X_test, Y_test, nodes, 2, 1e-4, 50000)

call_everthing()




