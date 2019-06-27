#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:42:41 2019

@author: adityaprasad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics


dataset_train = pd.read_csv('../data/spambase.data.txt',header = None)
dataset = dataset_train.values
np.random.seed(10)
np.random.shuffle(dataset)

for i in range(57):
    temp = dataset[:,i]
    dataset[:,i] = (temp - np.mean(temp)) / np.std(temp)
    
dataset = np.c_[np.ones(4601),dataset]

def linear_regression_Gradient_Descent(X,Y,error,accuracy):
    r,c = X.shape
    np.random.seed(42)
    W = np.random.randn(c) 
    lamb = 0.1
    for k in range(100):
        pred = (1/(1 + np.exp(-np.dot(X, W))))
        diff = Y - pred
        j = np.dot(X.T, diff)
        diag = np.diag(-pred * (1-pred))
        hessian = np.linalg.pinv(np.dot(np.dot(X.T,diag), X))
        W = W - lamb * np.dot(np.linalg.pinv(np.dot(np.dot(X.T,diag), X)),j)
    return W

def plot_roc_curve(y, y_test):
    p = np.linspace(min(y), max(y), 100)
    tpr = []
    fpr = []
    for j in p:
        y1 = []
        for i in range(len(y)):
            if(y[i] > j):
                y1.append(1)
            else:
                y1.append(0)
        y2 = np.array(y1)
        tp,fp,fn,tn = build_confusion_matrix(y2,y_test)
        t = tp / (tp + fn)
        f = fp / (fp + tn)
        tpr.append(t)
        fpr.append(f)
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("auc = ", -np.trapz(tpr,fpr))
    return -np.trapz(tpr,fpr)

def build_confusion_matrix(Y, y_test):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for i in range(len(Y)):
        if(y_test[i] == 1):
            if(Y[i] == 1):
                true_positive +=1
            if(Y[i] == 0):
                false_negative +=1
        if(y_test[i] == 0):
            if(Y[i] == 1):
                false_positive +=1
            if(Y[i] == 0):
                true_negative +=1
    return true_positive,false_positive,false_negative,true_negative


auc = 0
k_split = 4
k = [i for i in range(k_split)]
n = np.array_split(dataset, k_split)
error = 0
accuracy = 0
for i in range(k_split):
    data = n[i]
    X_test = data[:,:-1]
    y_test = data[:,-1]
    counter = 0
    for j in range(k_split):
        if(j!=i):
            if(counter == 0):
                data = n[j]
                counter +=1
            else:
                data = np.concatenate((data,n[j]),axis = 0)
    X = data[:,:-1]
    Y = data[:,-1]
    y_pred = linear_regression_Gradient_Descent(X,Y,error,accuracy)
    y = np.matmul(X_test,y_pred.T)
    #auc = auc + plot_roc_curve(y,y_test)
    y1 = []
    for i in range(len(y)):
        if(y[i] > (0.5)):
            y1.append(1)
        else:
            y1.append(0)
    y2 = np.array(y1)
    true_positive,false_positive,false_negative,true_negative = build_confusion_matrix(y2, y_test)
    #print(true_positive,"\t",false_positive,"\n",false_negative,"\t",true_negative,"\n")
    #build_confusion_matrix(y2, y_test)
    error = error + np.sum(np.square(y2 - y_test)) / len(y_test)
    print("error = ", np.sum(np.square(y2 - y_test)) / len(y_test))
    accuracy = accuracy + sklearn.metrics.accuracy_score(y_test, y2)
    print("accuracy = ", sklearn.metrics.accuracy_score(y_test, y2))
    
print(" testing error = ", error/k_split)
print("accuracy = ", accuracy/k_split)

"""
 H = np.matmul(W,X[i])
            G = 1 / (1 + np.exp(-H))
            a = X[i].reshape(1,c)
            diff = Y[i] - G
            hessian = np.linalg.pinv(np.dot(a.T,a) * (G * (1-G)))
            W = W + lamb * np.dot(a,hessian) * diff
            """
