import numpy as np

np.random.seed(42)

def generate_data(n_coins, m):
    pi = np.random.random(n_coins)
    pi = pi / np.sum(pi)
    q = np.random.random_sample(n_coins)
    data = []
    coins = np.arange(n_coins)
    for i in range(m):
        choice = np.random.choice(coins, 1, p = pi)
        data.append(np.random.binomial(1,q[choice], 10))
    print("pi_data = ", pi)
    print("q_data = ", q)
    return np.array(data)

def initialize_random(data, n_coins):
    Z = np.random.random((n_coins, data.shape[0]))
    Z = Z / np.sum(Z, axis=0)
    return Z

def M_step(Z, data, n_coins):
    pi = np.sum(Z,axis = 1) / data.shape[0]
    q = np.sum(Z * np.sum(data, axis = 1), axis = 1) / np.sum(Z * (data.shape[1]), axis = 1)
    return pi, q

def get_prob(data, q):
    prob_X = []
    for i in range(q.shape[0]):
        number_HT = []
        number_HT.append(np.count_nonzero(data, axis = 1))
        number_HT.append(np.count_nonzero(data == 0, axis = 1))
        number_HT = np.array(number_HT)
        prob = []
        prob.append(q[i])
        prob.append(1 - q[i])
        prob = np.array(prob)
        temp_prob_X = np.prod(np.power(prob.reshape(2,1), number_HT), axis=0)
        prob_X.append(temp_prob_X)
    return np.array(prob_X)


def E_step(pi, q, data, n_coins):
    prob_X = get_prob(data, q)
    Z = (pi.reshape(pi.shape[0],1) * prob_X) / np.sum(pi.reshape(pi.shape[0],1) * prob_X, axis = 0)
    return Z

def predict(data, n_coins, epochs):
    Z = initialize_random(data, n_coins)
    for i in range(epochs):
        pi, q = M_step(Z, data, n_coins)
        Z = E_step(pi, q, data, n_coins)
        if(i % 1000 == 0):
            print("pi = ", pi)
            print("q = ", q)


def run(n_coins, m, epochs):
    data = generate_data(n_coins, m)
    predict(data, n_coins, epochs)
    c = []

run(3,1000,10000)
