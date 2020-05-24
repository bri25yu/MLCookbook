"""
@author bri25yu

Note: this decision tree implementation assumes classification problem, not regression.
"""

import numpy as np

from classifiers.classifier import Classifier
from lib.vector_math import VectorMath as vm
from lib.data_handler import DataHandler as dh
from lib.boosting import Adaboost

class DecisionTree(Classifier):
    NUM_THRESHOLD_VALUES = 10
    THRESHOLD_EPS = 1e-5
    NUM_SUBSETS = 10
    GAIN_FUNC = vm.gini_purity
    MAX_DEPTH = 3

    def __init__(self, **kwargs):
        self.feature_dim = kwargs.get('feature_dim', None)
        assert self.feature_dim is not None, 'Must provide feature dimension!'
        self.left, self.right = None, None

        self.kwargs_to_pass = kwargs
        self.max_depth = kwargs.get('max_depth', DecisionTree.MAX_DEPTH)
        self.set_categorical_indices(kwargs.get('categorical_features_indices', []))
        self.gain_func = kwargs.get('gain_func', DecisionTree.GAIN_FUNC)
        self.num_subsets = kwargs.get('num_subsets', DecisionTree.NUM_SUBSETS)
        self.num_threshold_values = kwargs.get('num_threshold_values', DecisionTree.NUM_THRESHOLD_VALUES)

    def set_categorical_indices(self, categorical):
        self.categorical = categorical
        self.numerical = [i for i in range(self.feature_dim) if i not in self.categorical]

    def _argmax_gain(self, X, y, indices, values, is_cat):
        best = float('-inf'), None, None
        for i, feature_index in enumerate(indices):
            for val in values[i]:
                current_gain = self.gain_func(X[:, i], y, vm.get_split_fn(val, is_cat))
                if current_gain > best[0]: best = current_gain, feature_index, val
        return best

    def _get_feature_index_max(self, X, y):
        n, c = X[:, self.numerical], X[:, self.categorical]

        numerical = self._argmax_gain(n, y, self.numerical, vm.get_thresholds(n, self.NUM_THRESHOLD_VALUES, self.THRESHOLD_EPS), False)
        categorical = self._argmax_gain(c, y, self.categorical, vm.get_subsets(c, self.NUM_SUBSETS), True)

        best = categorical if categorical[0] > numerical[0] else numerical
        _, feature_index_max, split_val_max = best
        return feature_index_max, split_val_max, best is categorical

    def _fit_child(self, X, y):
        node = DecisionTree(**{**self.kwargs_to_pass, 'max_depth' : self.max_depth - 1})
        node.fit(X, y)
        return node

    def fit(self, X, y):
        bincounts = np.bincount(np.ravel(y))
        self.prediction = np.argmax(bincounts)
        self.error = 1 - (bincounts[self.prediction] / y.shape[0])

        if self.max_depth <= 0: return

        self.feature_index, self.split_val, self.is_categorical = self._get_feature_index_max(X, y)
        if self.split_val is not None:
            l, r = vm.split(X[:, self.feature_index], vm.get_split_fn(self.split_val, self.is_categorical))
            if not (l.shape[0] == 0 or r.shape[0] == 0):
                self.left, self.right = self._fit_child(X[l], y[l]), self._fit_child(X[r], y[r])
                self.weights = np.array([l.shape[0], r.shape[0]]) / (y.shape[0])

        self.prune()

    def predict(self, X):
        if not (self.left or self.right): return self.prediction * np.ones(X.shape[0])

        l, r = vm.split(X[:, self.feature_index], vm.get_split_fn(self.split_val, self.is_categorical))
        predictions = np.zeros(X.shape[0])
        predictions[l], predictions[r] = self.left.predict(X[l]), self.right.predict(X[r])
        return predictions

    def prune(self):
        if not (self.left or self.right): return self.error

        new_error = self.weights @ [self.left.prune(), self.right.prune()]
        if new_error > self.error: self.left, self.right = None, None
        else: self.error = new_error
        return self.error

    def toString(self, prepend=''):
        if not (self.left or self.right): return '%s(%s)' % (prepend, self.prediction)
        left = prepend + self.left.toString(prepend + '\t')
        right = prepend + self.right.toString(prepend + '\t')
        return '%s(feature: %s, split_val: %s, \n%s, \n%s)' % (prepend, self.feature_index, self.split_val, left, right)

class BaggedTree(DecisionTree):
    NUM_TREES = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_trees = kwargs.get('num_trees', BaggedTree.NUM_TREES)

    def initialize_trees(self):
        self.trees = [DecisionTree(**self.kwargs_to_pass) for _ in range(self.num_trees)]

    def fit(self, X, y):
        self.initialize_trees()
        for tree in self.trees:
            X_in, y_in, _, _ = dh.sample_with_replacement(X, y)
            tree.fit(X_in, y_in)

    def aggregate_predictions(self, preds):
        return np.array([np.argmax(np.bincount(p)) for p in np.array(preds).astype(int).T])

    def predict(self, X):
        return self.aggregate_predictions([t.predict(X) for t in self.trees])

class RandomForest(BaggedTree):

    class RandomDecisionTree(DecisionTree):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            subset_dim = kwargs.get('subset_dim', int(np.sqrt(self.feature_dim)))
            self.subset = np.random.choice(np.arange(self.feature_dim), size=subset_dim, replace=False)
            self.feature_dim = subset_dim
            self.set_categorical_indices([n for n, o in enumerate(self.subset) if o in self.categorical])
            self.kwargs_to_pass['feature_dim'] = self.feature_dim
            self.kwargs_to_pass['categorical_features_indices'] = self.categorical

        def fit(self, X, y):
            super().fit(X[:, self.subset], y)

        def predict(self, X):
            return super().predict(X[:, self.subset])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subset_dim = kwargs.get('subset_dim', int(np.sqrt(self.feature_dim)))

    def initialize_trees(self):
        self.trees = [self.RandomDecisionTree(**{**self.kwargs_to_pass, 'subset_dim' : self.subset_dim}) for _ in range(self.num_trees)]
        
class RandomForestAdaboost(Classifier):
    NUM_TRIALS = 10

    def __init__(self, rf, **kwargs):
        self.num_trials = kwargs.get('num_trials', RandomForestAdaboost.NUM_TRIALS)
        treesets = []
        for _ in range(self.num_trials):
            rf.initialize_trees()
            treesets.append(rf.trees)
        treesets = np.array(treesets)
        self.adaboost = Adaboost(treesets.T)

    def fit(self, X, y):
        self.adaboost.fit(X, y)

    def predict(self, X):
        return self.adaboost.predict(X)
