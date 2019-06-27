import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



def read_data():
    X = np.loadtxt("../data/Digits_dataset_20_percent_features.txt")
    Y = np.loadtxt("../data/Digits_dataset_20_percent_labels.txt")
    train_size = int(X.shape[0] * 0.8)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    data = np.c_[X,Y]
    np.random.shuffle(data)
    X_train = data[:train_size,:-1]
    Y_train = data[:train_size,-1]
    X_test = data[train_size:,:-1]
    Y_test = data[train_size:, -1]
    return X_train, Y_train, X_test, Y_test

def run():
    X_train, Y_train, X_test, Y_test = read_data()
    svm = SVC(C = 0.7, max_iter=100)
    svm.fit(X_train, Y_train)
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)
    print("training accuracy = ", accuracy_score(Y_train, y_pred_train))
    print("testing accuracy = ", accuracy_score(Y_test, y_pred_test))

run()

