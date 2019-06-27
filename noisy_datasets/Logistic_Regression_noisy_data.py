import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
 
class LogisticRegression_Exam:
    def read_data(self):
        X_train = np.load('../data/x_train.npy')
        Y_train = np.load('../data/y_train.npy')
        X_test_noisy = np.load('../data/x_test_noisy.npy')
        Y_test = np.load('../data/y_test.npy')
 
        return X_train, Y_train, X_test_noisy, Y_test
 
    def run(self):
        X_train, Y_train, X_test_noisy, Y_test  = self.read_data()
        print("done_reading")
        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        clf.fit(X_train, Y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test_noisy)
 
        print("Training accuracy = ", accuracy_score(Y_train, y_pred_train))
        print("Testing accuracy = ", accuracy_score(Y_test, y_pred_test))
 
 
 
logistic = LogisticRegression_Exam()
logistic.run()