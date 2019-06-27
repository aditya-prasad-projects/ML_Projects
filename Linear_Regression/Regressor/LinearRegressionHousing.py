#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:15:49 2019

@author: adityaprasad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('../../data/housing_train.txt',sep = '\s+',header = None)
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, 13].values

X_train = np.c_[np.ones(433), X_train]

dataset_test = pd.read_csv('../../data/housing_test.txt',sep = '\s+', header = None)
X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, 13].values

X_test = np.c_[np.ones(74),X_test]

X_trainT = np.transpose(X_train)
#X = np.linalg.pinv(np.dot(X_train, X_trainT))
V = np.dot(np.dot(np.linalg.pinv(np.dot(X_trainT, X_train)), X_trainT),y_train)

y_pred = np.sum(np.multiply(X_test,V),axis = 1)
print(np.sum(np.square(y_pred - y_test)) / len(y_test))








