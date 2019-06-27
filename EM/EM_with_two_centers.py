import pandas as pd
import numpy as np

from scipy.stats import multivariate_normal

np.random.seed(10)


def read_data(filename):
    data = pd.read_csv(filename, sep = " ", header = None)
    return data.values

def calculate_prob(mu0, mu1,sigma0, sigma1,data):
    prob_X = []
    variance0 = data - mu0
    variance1 = data - mu1
    denominatorpi = np.power((2 * np.pi), (data.shape[1] / 2))
    denominator0 = np.log2(1 / (np.sqrt(np.linalg.det(sigma0)) * denominatorpi))
    denominator1 = np.log2(1 / (np.sqrt(np.linalg.det(sigma1)) * denominatorpi))
    sigmaInverse0 = np.linalg.pinv(sigma0)
    sigmaInverse1 = np.linalg.pinv(sigma1)
    exp0 = np.sum(np.multiply(np.dot(variance0, sigmaInverse0), variance0), axis = 1)
    exp1 = np.sum(np.multiply(np.dot(variance1, sigmaInverse1), variance1), axis = 1)
    pxy0 = denominator0 * exp0
    pxy1 = denominator1 * exp1
    prob_X.append(pxy0)
    prob_X.append(pxy1)
    return np.array(prob_X)

def E_step(Z, mu0, mu1,sigma0, sigma1, pi0, pi1, data):
    prob_X = []
    prob_X.append(multivariate_normal.pdf(data, mean=mu0, cov=sigma0))
    prob_X.append(multivariate_normal.pdf(data, mean=mu1, cov=sigma1))

    prob_X = np.array(prob_X)
    gaussian_mixture = np.array([[pi0], [pi1]])
    Z = (prob_X * gaussian_mixture) / np.sum((prob_X * gaussian_mixture),axis = 0)
    return Z

def M_step(Z, data):
    mu0 = np.sum(Z[0] * data.T, axis = 1) / np.sum(Z[0])
    mu1 = np.sum(Z[1] * data.T, axis = 1) / np.sum(Z[1])
    #mu0, mu1 = np.dot(Z, data) / np.sum(Z, axis = 1)
    pi0, pi1 = np.sum(Z,axis = 1) / data.shape[0]
    sigma0 = np.dot((Z[0] * (data - mu0).T), (data - mu0)) / np.sum(Z[0])
    sigma1 = np.dot((Z[1] * (data - mu1).T), (data - mu1)) / np.sum(Z[1])
    return mu0, mu1, sigma0, sigma1, pi0, pi1

def calculate_cost(Z, mu0, mu1, sigma0, sigma1, pi0, pi1, data):
    gaussian_mixture = np.array([[pi0], [pi1]])
    log_probs = []
    log_probs.append(multivariate_normal.pdf(data, mean=mu0, cov=sigma0))
    log_probs.append(multivariate_normal.pdf(data, mean=mu1, cov=sigma1))
    log_probs = np.array(log_probs)
    cost = -(np.sum((Z * log_probs) + (Z * np.log2(gaussian_mixture))))
    return cost

def initialize_random(data):
    Z = np.random.random((2, data.shape[0]))
    Z = Z / np.sum(Z, axis = 0)
    return Z

def run(epochs):
    data = read_data("../data/2-gaussian.txt")
    Z = initialize_random(data)
    for i in range(epochs):
        mu0, mu1, sigma0, sigma1, pi0, pi1 = M_step(Z, data)
        Z = E_step(Z, mu0, mu1,sigma0, sigma1, pi0, pi1, data)
        if(i % 10 == 0):
            cost = calculate_cost(Z, mu0, mu1, sigma0, sigma1, pi0, pi1, data)
            print("cost = ", cost)
            #print(Z)
            print("mu0 - ", mu0)
            print("mu1 = ", mu1)
            print("sigma0 = ",sigma0)
            print("sigma1 = ", sigma1)
            print("pi0 = ", pi0)
            print("pi1 = ", pi1)

run(100)