"""
@author bri25yu
"""

import numpy as np
from scipy.special import expit

class GradientDescent:
    name = 'gd'

    def __init__(self, X, y, eps, lmbda, num_iterations, cost_step=100):
        # X is nxd, y is nx1, eps is a scalar
        self.X, self.y, self.eps, self.lmbda = X, y, eps, lmbda

        self.weights = np.random.random((self.X.shape[1], 1))
        self.num_iterations, self.cost_step = num_iterations, cost_step
        self.cost_iterations, self.costs = [], []

    def run(self):
        pass

    def update_weights(self):
        pass

    def cost(self):
        pass

    def classify(self, X):
        return np.round(expit(X @ self.weights)).astype('int64')

class BatchGradientDescent(GradientDescent):
    name = 'bgd'

    def run(self, verbose=False):
        for i in range(self.num_iterations):
            self.update_weights()
            if not i % self.cost_step:
                self.costs.append(self.cost())
                self.cost_iterations.append(i)
            if verbose and not i % 1000: print('Finished %s / %s iterations' % (i, self.num_iterations))

    def update_weights(self):
        first = (1 - self.eps * self.lmbda) * self.weights
        second = self.eps * self.X.T @ (self.y - expit(self.X @ self.weights))
        self.weights = first + second
        self.weights = self.weights

    def cost(self):
        regularization = (self.lmbda / 2) * np.linalg.norm(self.weights)
        sig = expit(self.X @ self.weights)
        first = self.y.T @ np.log(sig)
        second = (1 - self.y).T @ (np.log(1 - sig))
        return (regularization - first - second)[0][0]

class StochasticGradientDescent(GradientDescent):
    name = 'sgd'

    def __init__(self, X, y, eps, lmbda, num_iterations, cost_step=1000):
        super().__init__(X, y, eps, lmbda, num_iterations, cost_step)
        self.i = 0

    def run(self, verbose=False):
        for i in range(self.num_iterations):
            self.update_weights()
            if not i % self.cost_step:
                self.costs.append(self.cost())
                self.cost_iterations.append(i)
                if verbose: print('Finished %s / %s iterations' % (i, self.num_iterations))

    def update_weights(self):
        first = (1 - self.eps * self.lmbda) * self.weights
        second = self.eps * (self.y[self.i] - expit(self.X[self.i] @ self.weights)) * self.X[self.i]
        second = np.reshape(second, (second.shape[0], 1))
        self.weights = first + second
        self.i = (self.i + 1) % self.X.shape[0]

    def cost(self):
        regularization = (self.lmbda / 2) * np.linalg.norm(self.weights)
        sig = expit(self.X @ self.weights)
        first = self.y.T @ np.log(sig)
        second = (1 - self.y).T @ (np.log(1 - sig))
        return (regularization - first - second)[0][0]

class EpsilonStochasticGradientDescent(StochasticGradientDescent):
    name = 'esgd'

    def __init__(self, X, y, delta, lmbda, num_iterations, cost_step=1000):
        super().__init__(X, y, None, lmbda, num_iterations, cost_step)
        self.delta = delta
        self.indices = []
        def next_i():
            if not self.indices:
                self.indices = list(range(X.shape[0]))
                np.random.shuffle(self.indices)
            return self.indices.pop(0)
        self.next_i = next_i

    def update_weights(self):
        self.eps = self.delta / (self.i + 1)
        first = (1 - self.eps * self.lmbda) * self.weights
        second = self.eps * (self.y[self.i] - expit(self.X[self.i] @ self.weights)) * self.X[self.i]
        second = np.reshape(second, (second.shape[0], 1))
        self.weights = first + second
        self.i = self.next_i()
