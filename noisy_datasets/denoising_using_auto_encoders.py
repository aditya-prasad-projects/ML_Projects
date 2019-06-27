import numpy as np
import tensorflow as tf

X_train = np.load('../data/x_train.npy')
Y_train = np.load('../data/y_train.npy')
X_test_noisy = np.load('../data/x_test_noisy.npy')
Y_test = np.load('../data/y_test.npy')


print(X_train[0].shape)
inputs  = tf.keras.layers.Input(shape=(784,))              # 28*28 flatten
layer_1  = tf.keras.layers.Dense(128, activation='relu') (inputs)


encoded = layer_1
dec_fc  = tf.keras.layers.Dense(784, activation='sigmoid') # decompress to 784 pixels
decoded = dec_fc(encoded)
autoencoder = tf.keras.models.Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

X_test  = (X_test_noisy)


autoencoder.fit(X_train, X_train, epochs = 20, batch_size=128)


X_test_decoded = autoencoder.predict(X_test)
np.savetxt("./X_test_decoded.txt", X_test_decoded)
