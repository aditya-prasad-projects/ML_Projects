import numpy as np

np.random.seed(1)
X = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
             dtype=np.float32)

def initialize_weights_bias():
    W1 = np.random.normal(size=(3,8))
    b1 = np.zeros((3,1))
    W2 = np.random.normal(size=(8,3))
    b2 = np.zeros((8,1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def forward_propagation(parameters, X):
    Z1 = np.matmul(parameters["W1"], X) + parameters["b1"]
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.matmul(parameters["W2"], A1) + parameters["b2"]
    A2 = 1 / (1 + np.exp(-Z2))
    activation = {"A1":A1, "A2":A2}
    return activation, parameters

def calculate_cost(activation, Y):
    m = activation["A2"].shape[1]
    cost = (-1/m) * np.sum(np.multiply(Y,np.log(activation["A2"].T)) + np.multiply((1-Y), np.log(1-activation["A2"].T)))
    return cost

def backward_propagation(activation, X, Y, parameters, learning_rate):
    m = activation["A2"].shape[1]
    dZ2 = activation["A2"] - Y
    dW2 = (1/m)  * np.matmul(dZ2, activation["A1"].T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.matmul(parameters["W2"].T, dZ2), np.multiply(activation["A1"], (1-activation["A1"])))
    dW1 = (1/m) * np.matmul(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims = True)
    parameters["W1"] = parameters["W1"] - (learning_rate * dW1)
    parameters["b1"] = parameters["b1"] - (learning_rate * db1)
    parameters["W2"] = parameters["W2"] - (learning_rate * dW2)
    parameters["b2"] = parameters["b2"] - (learning_rate * db2)
    return parameters

def gradient_descent(X,Y,epochs, learing_rate):
    parameters = initialize_weights_bias()
    for i in range(epochs):
        activation,parameters = forward_propagation(parameters, X)
        cost = calculate_cost(activation, Y)
        parameters = backward_propagation(activation,X,Y,parameters,learing_rate)
        if(i % 5000 == 0):
            print("cost = ", cost)
    print(np.argmax(activation["A2"], axis = 1))

gradient_descent(X.T,X.T,30000, 0.5)








