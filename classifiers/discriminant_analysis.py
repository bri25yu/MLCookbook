"""
@author bri25yu
"""

import numpy as np

from classifiers.classifier import Classifier
from lib.vector_math import VectorMath as vm
from lib.data_handler import DataHandler as dh

class DiscriminantAnalysis(Classifier):
    
    def calculateMLE(self, X):
        return vm.mean(X), vm.cov(X)

    def fit(self, X, y):
        X = dh.group_by_label(X, y)
        for key in X.keys():
            d = {}
            d['prior'] = X[key].shape[0] / y.shape[0]
            d['mean'], d['cov'] = self.calculateMLE(X[key])
            X[key] = d
        self.mle = X

class LinearDA(DiscriminantAnalysis):

    def __init__(self, lmbda):
        self.lmbda = lmbda
    
    def fit(self, X, y):
        super().fit(X, y)
        self.cov = vm.add_cov_matrices([c.pop('cov') for c in self.mle.values()])
        self.cov = vm.add_lambda_I(self.cov, self.lmbda)

    def classify_lda_datum(self, class_info, x):
        get_particular = lambda c: class_info[c][0] @ x + class_info[c][1]
        return max(class_info.keys(), key = get_particular)

    def predict(self, X):
        class_info = {}
        cov_inv = np.linalg.inv(self.cov)
        for key, d in self.mle.items():
            mean, prior = d['mean'], d['prior']
            mean_cov_inv = mean.T @ cov_inv
            class_info[key] = (mean_cov_inv, -0.5 * mean_cov_inv @ mean + np.log(prior))
        return np.array([self.classify_lda_datum(class_info, x) for x in X])

class QuadraticDA(DiscriminantAnalysis):

    def __init__(self, lmbdas):
        self.lmbdas = lmbdas

    def fit(self, X, y):
        super().fit(X, y)
        for c, d in self.mle.items():
            d['cov'] = vm.add_lambda_I(d['cov'], self.lmbdas[c])

    def classify_qda_datum(self, ci, x):
        get_particular = lambda c: -0.5 * (x - ci[c][0]).T @ ci[c][1] @ (x - ci[c][0]) + ci[c][2]
        class_vals = [[c, get_particular(c)] for c in ci.keys()]
        return max(class_vals, key=lambda p: p[1])[0]

    def predict(self, X):
        class_info = {}
        for key, d in self.mle.items():
            mean, prior, cov = d['mean'], d['prior'], d['cov']
            cov_inv = np.linalg.inv(cov)
            class_info[key] = mean, cov_inv, -0.5 * np.linalg.slogdet(cov)[1] + np.log(prior)
        return np.array([self.classify_qda_datum(class_info, x) for x in X])

