#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:10:32 2019

@author: adityaprasad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('../data/perceptronData.txt',sep = '\s+', header = None)
dataset = dataset_train.values


for i in range(len(dataset)):
    if(dataset[i,-1] == -1):
        dataset[i] = dataset[i] * -1
dataset = np.c_[np.ones(1000),dataset]

W = np.random.randn(5)

X = dataset[:,:-1]
Y = dataset[:,-1]

def get_M(X, W):
    M = []
    for i in range(len(X)):
        if(np.dot(X[i],W) < 0):
            M.append(X[i])
    return np.array(M)

M = get_M(X, W)
lamb = 10
j = 0

while(len(M) > 0):
    
    M = get_M(X,W)
    W = W + (lamb * np.sum(M,axis = 0))
    print("iteration = ", j, "Total_error = ", len(M))
    j +=1
    
"""

while(len(M) > 10):
    if(j > 1000):
        lamb = 0.000001
    if(lamb > 2000):
        lamb = 0.0000001
    if(lamb > 3000):
        lamb = 0.00000001
    M = get_M(X,W)
    for i in range(len(M)):
       W = W + (lamb * np.transpose(M[i]))
    print("iteration = ", j, "Total_error = ", len(M))
    j +=1
    """

    

print(W)
print(W / W[0])
    
 

    
    

