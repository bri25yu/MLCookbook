"""
@author bri25yu
"""

import numpy as np
from collections import Counter

from classifiers.classifier import Classifier
from lib.data_handler import DataHandler as dh

class Adaboost(Classifier):

    def __init__(self, classifier_sets):
        self.weights = []
        self.classifier_sets = classifier_sets
        self.classifiers = []

    def fit_classifier(self, classifier_set, X, y, data_weights):
        classifier, error = None, float('inf')

        for c in classifier_set:
            selected, not_selected = dh.oob(X.shape[0])
            X_in, y_in, X_oob, y_oob = X[selected], y[selected], X[not_selected], y[not_selected]
            c.fit(X_in, y_in)

            e = (((c.predict(X_oob) != y_oob) @ data_weights[not_selected]) * y.shape[0]) / not_selected.shape[0]
            if e < error: classifier, error = c, e

        incorrect = classifier.predict(X) != y
        classifier_weight = 0.5 * np.log((1 - error) / error)
        data_weights = data_weights * np.exp(classifier_weight * (incorrect * 2 - 1))
        data_weights = data_weights / np.sum(data_weights)

        self.weights.append(classifier_weight)
        self.classifiers.append(classifier)

        return data_weights

    def fit(self, X, y):
        y = np.ravel(y)
        data_weights = np.ones(X.shape[0]) * (1 / X.shape[0])

        for classifer_set in self.classifier_sets:
            data_weights = self.fit_classifier(classifer_set, X, y, data_weights)

        self.weights = np.array(self.weights)

    def aggregate_predictions(self, preds):
        predictions = []
        for p in np.array(preds).astype(int).T:
            c = Counter()
            for p_val, w in zip(p, self.weights):
                c[p_val] += w
            predictions.append(c.most_common(1)[0][0])
        return np.array(predictions)

    def predict(self, X):
        preds = [c.predict(X) for c in self.classifiers]
        return self.aggregate_predictions(preds)
