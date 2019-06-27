import tensorflow as tf
import numpy as np

X = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], dtype=np.float32)

def initialize_parameters():
    W1 = tf.get_variable("W1", [8,3], initializer= tf.random_normal_initializer,dtype=tf.float32)
    b1 = tf.get_variable("b1", [1,3], initializer=tf.zeros_initializer,dtype=tf.float32)
    W2 = tf.get_variable("W2", [3,8], initializer= tf.random_normal_initializer,dtype=tf.float32)
    b2 = tf.get_variable("b2", [1,8], initializer=tf.zeros_initializer,dtype=tf.float32)
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    return parameters

np.random.seed(1)


def encoder():
    parameters = initialize_parameters()

    A1 = tf.nn.sigmoid(tf.matmul(X,parameters["W1"]) + parameters["b1"])
    A2 = tf.nn.sigmoid(tf.matmul(A1, parameters["W2"] + parameters["b2"]))
    return A2

def loss(y_pred, Y):
    cost = tf.losses.log_loss(Y,y_pred)
    return cost


def run(X_train):
    X = tf.placeholder(tf.float32, [8,None])
    output = encoder()
    cost = loss(output, X_train)
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {X:X_train}
        for i in range(30000):
            _,loss_val = sess.run([optimizer,cost], feed_dict)
            if(i % 5000 == 0):
                print(loss_val)
        out = sess.run(output,feed_dict)
        print(np.argmax(out,axis= 1))
run(X)


