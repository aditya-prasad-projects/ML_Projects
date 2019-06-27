#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:55:53 2019

@author: adityaprasad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('../../data/housing_train.txt',sep = '\s+',header = None)
dataset = dataset_train.values
dataset_test = pd.read_csv('../../data/housing_test.txt',sep = '\s+', header = None)
dataset = np.append(dataset, dataset_test.values, axis = 0)

#Normalize the data
for i in range(13):
    temp = dataset[:,i]
    dataset[:,i] = temp - np.mean(temp)
b = np.mean(dataset[:,-1])
#Split train and test data
X_train = dataset[:433, :-1]
y_train = dataset[:433, 13]
X_test = dataset[433:, :-1]
y_test = dataset[433:, 13]

X_train = np.c_[np.full(433,b), X_train]

X_test = np.c_[np.full(74,b),X_test]

X_trainT = np.transpose(X_train)
j = 0.0
mini = 30
mini1 = 30

for i in np.arange(0.001,1,0.001):
    W = np.dot(np.dot(np.linalg.inv(np.add(np.dot(X_trainT, X_train), 0 * np.identity(14))), X_trainT),y_train)
    y_pred = np.matmul(X_test,W)
    y_pred1 = np.matmul(X_train, W)
    b = np.sum(np.square(y_pred1 - y_train)) / len(y_train)
    a = (np.sum(np.square(y_pred - y_test)) / len(y_test))
    if(a <= mini):
        mini = a
        j = i
    if(b <= mini1):
        mini1 = b
        
        
        
print("mse = ", mini, " alpha = ", j)
print(mini1)

        


