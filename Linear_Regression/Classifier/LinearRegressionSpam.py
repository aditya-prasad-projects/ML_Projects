#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:17:56 2019

@author: adityaprasad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics


dataset_train = pd.read_csv('../../data/spambase.data.txt',header = None)
dataset = dataset_train.values
np.random.shuffle(dataset)

for i in range(57):
    temp = dataset[:,i]
    dataset[:,i] = (temp - np.mean(temp)) / np.std(temp)
    
dataset = np.c_[np.ones(4601),dataset]

def linear_regression(X,Y):
    X_trainT = np.transpose(X)
#X = np.linalg.pinv(np.dot(X_train, X_trainT))
    
    W = np.dot(np.dot(np.linalg.pinv(np.dot(X_trainT, X)), X_trainT),Y)
    return W

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
    y_pred = linear_regression(X,Y)
    y = np.sum(np.multiply(X_test,y_pred),axis = 1)
    y1 = []
    for i in range(len(y)):
        if(y[i] > (0.411)):
            y1.append(1)
        else:
            y1.append(0)
    y2 = np.array(y1)
    error = error + np.sum(np.square(y2 - y_test)) / len(y_test)
    print("error = ",error)
    accuracy = accuracy + sklearn.metrics.accuracy_score(y_test, y2)
    
    print("accuracy = ",accuracy)
    
    
print(error/k_split)
print(accuracy/k_split)