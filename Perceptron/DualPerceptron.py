import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(42)
from sklearn.metrics import accuracy_score



class Perceptron:

    def __init__(self, filename, kernel):
        self.filename = filename
        self.kernel = kernel

    def read(self):
        dataset_train = pd.read_csv(self.filename, sep='\s+', header=None)
        dataset = dataset_train.values
        self.X = dataset[:,:-1]
        self.Y = dataset[:,-1]
        self.original_Y = np.copy(self.Y)
        self.count_vector = np.where(self.Y == -1, 0, 0)


    def transfer_to_positive(self):
        negative_indices = np.where(self.Y == -1)[0]
        self.X[negative_indices] *= -1
        self.Y[negative_indices] *= -1

    def add_bias(self):
        self.X = np.c_[np.ones(self.X.shape[0]), self.X]
        self.W = np.random.randn(self.X.shape[1])


    def get_dot_product(self, x):
        return np.sum(np.multiply(self.X, x), axis = 1)

    def get_RBF(self, x):
        return np.exp((-(np.linalg.norm(self.X - x, axis=1)) / (2 * (0.1 ** 2))))
        #return np.exp(np.linalg.norm(self.X - x, ord=2, axis=1)) / (2 * ((0.1 ** 2)))

    def get_kernel(self,x):
        if(self.kernel == "dot"):
            return self.get_dot_product(x)
        elif(self.kernel == "RBF"):
            return self.get_RBF(x)

    def check_mistakes(self,x,y):
        dot_product = self.get_kernel(x)
        multiply_count = np.sum(np.multiply(self.count_vector, dot_product)) * y
        return multiply_count

    def get_mistakes(self):
        get_line = []
        for i in range(self.X.shape[0]):
            get_line.append(self.check_mistakes(self.X[i], self.Y[i]))
        get_line = np.array(get_line)
        return np.where(get_line <= 0)[0]

    def run(self):
        self.read()
        self.add_bias()
        self.transfer_to_positive()


        mistakes = self.get_mistakes()
        j = 0
        while(mistakes.size > 0):
            print("iteration = ", j, "total mistakes = ", mistakes.size)
            self.count_vector[mistakes] += 1
            mistakes = self.get_mistakes()
            j += 1
            if(j == 100):
                break

filenames = ['../data/twoSpirals.txt', '../data/perceptronData.txt']
dual_perceptron = Perceptron(filenames[1], "RBF")
dual_perceptron.run()




