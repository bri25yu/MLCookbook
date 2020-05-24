"""
@author bri25yu
"""

import numpy as np
import pandas as pd
import os

from multiprocessing import Pool
from itertools import repeat, product
from collections import OrderedDict

from lib.vector_math import VectorMath as vm
from lib.data_handler import DataHandler as dh
from lib.visualize import Visualization as vs

class KFoldCrossValidation:
    NUM_TRIALS = 3
    K = 5
    NUM_PROCESSES = 20

    @staticmethod
    def parse_kfold_kwargs(kwargs):
        num_trials = kwargs.get('num_trials', KFoldCrossValidation.NUM_TRIALS)
        k = kwargs.get('k', KFoldCrossValidation.K)
        num_processes = kwargs.get('num_processes', KFoldCrossValidation.NUM_PROCESSES)
        return num_trials, k, num_processes

    @staticmethod
    def cross_validate(td, tl, vd, vl, classifier, seed=None):
        if seed is not None: np.random.seed(seed)
        classifier.fit(td, tl)
        return vm.get_loss(vl, np.reshape(classifier.predict(vd), vl.shape))

    @staticmethod
    def kFoldCrossValidate(data, labels, classifiers, k=None):
        if k is None: k = KFoldCrossValidation.K
        error = 0
        for i, classifier in enumerate(classifiers):
            if i%k == 0: ds, ls = dh.partitionIntoK(data, labels, k)
            error += KFoldCrossValidation.cross_validate(*dh.combine(ds, ls, i % k), classifier)
        return error / len(classifiers)

    @staticmethod
    def kFoldCrossValidate_multitask(data, labels, classifier, **kwargs):
        num_trials, k, num_processes = KFoldCrossValidation.parse_kfold_kwargs(kwargs)
        
        args = []
        for i in range(num_trials * k):
            if i%k == 0: ds, ls = dh.partitionIntoK(data, labels, k)
            args.append(list(dh.combine(ds, ls, i%k)) + [classifier])

        with Pool(num_processes) as pool:
            results = pool.starmap(KFoldCrossValidation.cross_validate, args)
        return np.mean(results)

class HyperparameterTuning:
    NUM_PROCESSES = 20

    class Hyperparameter:
        EXPONENTIAL_MAPPING_FN = np.exp
        EXPONENTIAL_BASE_10_MAPPING_FN = lambda v: np.pow(v, 10)
        IDENTITY_MAPPING_FN = lambda x: x
        NUM_VALUES = 10
        IS_INTEGRAL = False

        def __init__(self, name, **kwargs):
            self.name = name
            self.is_integral = kwargs.get('is_integral', ht.Hyperparameter.IS_INTEGRAL)
            self.start = kwargs.get('start', None)
            self.end = kwargs.get('end', None)
            self.num_values = kwargs.get('num_values', ht.Hyperparameter.NUM_VALUES)
            self.map_fn = kwargs.get('map_fn', ht.Hyperparameter.IDENTITY_MAPPING_FN)
            self.values = self.premap_values = kwargs.get('values', [])
            assert (self.start and self.end and self.num_values) or self.values, 'Must provide values or (start, end, num_values).'
            if not self.values: self.generate_values(self.start, self.end, self.num_values, self.map_fn)

        def generate_values(self, start, end, num_values, map_fn=None):
            if map_fn is None: map_fn = self.map_fn
            dtype = int if self.is_integral else float
            self.premap_values = np.linspace(start, end, num_values, dtype=dtype)
            self.values = list(set(map(map_fn, self.premap_values)))

    class HyperparameterFrame:

        def __init__(self, hyperparameters):
            self.initialize(hyperparameters)

        def initialize(self, hyperparameters):
            self.hyperparameters = hyperparameters
            self.names = [hyppm.name for hyppm in hyperparameters]
            self.values = [hyppm.values for hyppm in hyperparameters]
            self.premap_values = [hyppm.premap_values for hyppm in hyperparameters]
            self.indices = dict((n, i) for i, n in enumerate(self.names))

        @property
        def dim(self):
            return len(self.hyperparameters)

        @property
        def shape(self):
            return [len(vals) for vals in self.values]

        def append(self, hyperparameter):
            self.hyperparameters.append(hyperparameter)
            self.initialize(self.hyperparameters)

        def index(self, name):
            return self.indices.get(name, -1)

    def __init__(self, data, labels, **kwargs):
        self.data, self.labels = data, labels
        self.num_processes = kwargs.get('num_processes', HyperparameterTuning.NUM_PROCESSES)
        self.results_history = []
    
    def post_processing(self, frame, results):
        self.errors, self.values = [p[0] if not np.isnan(p[0]) else float('inf') for p in results], [p[1] for p in results]
        min_error, min_values = results[np.argmin(self.errors)]
        self.errors = np.reshape(np.array([p[0] for p in results]), frame.shape)
        self.results_history.append([self.errors, self.values, min_error, min_values])
        return min_error, min_values

    def grid_search(self, frame, fn, output=True):
        self.frame = frame

        num_trials = np.prod(frame.shape)
        def fn_wrapper(i, v):
            result = fn(**OrderedDict(zip(frame.names, v)))
            if output and (i + 1) % 100 == 0: print('Finished %s / %s' % (i + 1, num_trials))
            return result

        results = [fn_wrapper(i, v) for i, v in enumerate(product(*frame.values))]
        return self.post_processing(frame, results)

    @staticmethod
    def grid_search_multitask_fn(seed, data, labels, classifier_class, values, names, kwargs):
        np.random.seed(seed)
        kwargs.update(zip(names, values))
        classifiers = [classifier_class(**kwargs) for _ in range(KFoldCrossValidation.K)]
        return KFoldCrossValidation.kFoldCrossValidate(data, labels, classifiers), values

    def grid_search_multitask(self, frame, classifier_class, classifier_kwargs):
        grid = product(*frame.values)
        seeds = np.random.randint(2**31 - 1, size=np.prod(frame.shape))
        with Pool(self.num_processes) as pool:
            results = pool.starmap(
                                    HyperparameterTuning.grid_search_multitask_fn,
                                    zip (
                                            seeds,
                                            repeat(self.data),
                                            repeat(self.labels),
                                            repeat(classifier_class),
                                            grid,
                                            repeat(frame.names),
                                            repeat(classifier_kwargs)
                                        )
                                )
        return self.post_processing(frame, results)

    def random_search(self):
        pass

    def evolution(self):
        pass

    def visualize2D(self, p1, p2, path='', show=True):
        i1, i2 = self.frame.index(p1), self.frame.index(p2)
        indices_to_aggregate = [i for i in range(self.frame.dim) if i != i1 and i != i2]
        errors = pd.DataFrame(data=np.mean(self.errors, axis=tuple(indices_to_aggregate)))
        errors = errors.rename(index=dict(zip(range(self.frame.shape[0]), ['%s_%s' % (p1, val) for val in self.frame.premap_values[i1]])))
        errors = errors.rename(columns=dict(zip(range(self.frame.shape[1]), ['%s_%s' % (p2, val) for val in self.frame.premap_values[i2]])))
        save_name = os.path.join(path, 'visualize2D_%s_%s' % (p1, p2))
        vs.visualize2D(errors, save_name=save_name, show=show)

ht = HyperparameterTuning
