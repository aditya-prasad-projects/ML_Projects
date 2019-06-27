 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:37:17 2019

@author: adityaprasad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read and combine test and training data
dataset_train = pd.read_csv('../../data/housing_train.txt',sep = '\s+',header = None)
dataset = dataset_train.values
dataset_test = pd.read_csv('../../data/housing_test.txt',sep = '\s+', header = None)
dataset = np.append(dataset, dataset_test.values, axis = 0)

#Normalize the data
for i in range(13):
    temp = dataset[:,i]
    dataset[:,i] = (temp - np.mean(temp)) / np.std(temp)

#Split train and test data
X_train = dataset[:433, :-1]
y_train = dataset[:433, 13]
X_test = dataset[433:, :-1]
y_test = dataset[433:, 13]


X_train = np.c_[np.ones(433), X_train]
X_test = np.c_[np.ones(74),X_test]

r,c = X_train.shape
np.random.seed(42)
W = np.random.randn(c) 
lamb = 0.0001
for k in range(101):
    for i in range(r):
        H = np.sum(np.multiply(X_train[i],W))
        diff = H - y_train[i]
        W = W - (X_train[i] * (lamb * diff))
        
    
    if(k%100 == 0):
        y_pred_train = np.sum(np.multiply(X_train,W),axis = 1)
        print("Training Stochastic = ",np.sum(np.square(y_pred_train - y_train)) / len(y_train))
        y_pred = np.sum(np.multiply(X_test,W),axis = 1)
        print("Test Stochastic = ",np.sum(np.square(y_pred - y_test)) / len(y_test))

        
