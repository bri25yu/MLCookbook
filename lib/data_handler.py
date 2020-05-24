"""
@author bri25yu
"""

from scipy import io, stats
import numpy as np
import pandas as pd
import csv
from collections import defaultdict

from lib.vector_math import VectorMath as vm

np.random.seed(1234)

class DataHandler:

    @staticmethod
    def load_data_csv(data_path):
        assert data_path.endswith('.csv'), 'Loading a not .csv file as a .csv file!'
        data = pd.read_csv(data_path, header=None)
        data = data.to_numpy()
        return data

    @staticmethod
    def load_data_mat(data_path):
        assert data_path.endswith('.mat'), 'Loading a not .mat file as a .mat file!'
        data = io.loadmat(data_path)
        return data

    @staticmethod
    def group_by_label(data, labels):
        g = defaultdict(list)
        for l, d in zip(labels, data):
            g[l[0]].append(d)
        for key in g:
            g[key] = np.array(g[key])
        return g

    @staticmethod
    def find_categorical(data):
        """
        Returns a list of the columns of the input data that correspond to categorical values.

        Parameters
        ----------
        data : np.ndarray
            An (n, d)-shaped array of data.

        Returns
        -------
        categorical : list
            A list of the columns of the input data that correspond to categorical values.

        """
        categorical = []
        for i in range(data.shape[1]):
            isCategorical = False
            for j in range(data.shape[0]):
                try:
                    float(data[j, i])
                except ValueError:
                    isCategorical = True
                    break
                except:
                    pass
            if isCategorical: categorical.append(i)
        return categorical

    @staticmethod
    def discretize_categorical(data, categorical):
        """
        Parameters
        ----------
        data : np.ndarray
            An (n, d)-shaped array.
        """
        for i in categorical:
            values = set(data[:, i])
            mappings = dict(zip(values, range(len(values))))
            data[:, i] = list(map(lambda k: mappings[k], data[:, i]))
        return data

    @staticmethod
    def isnan(val):
        try: return np.isnan(val)
        except: return False

    @staticmethod
    def get_not_nan(arr):
        """
        Parameters
        ----------
        arr : np.ndarray
            An (n, 1)-shaped array.
        """
        return arr[np.where(list(map(lambda v: not DataHandler.isnan(v), arr)))[0]]

    @staticmethod
    def get_filler_value(data, is_categorical=False):
        """

        Parameters
        ----------
        data : np.ndarray
            An (n, 1)-shaped array.

        """
        if is_categorical:
            return stats.mode(data).mode[0]
        return np.median(data)

    @staticmethod
    def fill_missing_values(data, categorical_features=None, missing_value_fn=None):
        """

        Parameters
        ----------
        data : np.ndarray
            An (n, d)-shaped array.

        """
        if categorical_features is None: categorical_features = DataHandler.find_categorical(data)
        if missing_value_fn is None:
            missing_value_fn = DataHandler.isnan

        filler_values = [
            DataHandler.get_filler_value(data[:, i], i in categorical_features)
            for i in range(data.shape[1])]

        for data_point in data:
            for feature_index in range(data.shape[1]):
                val = data_point[feature_index]
                if missing_value_fn(val):
                    data_point[feature_index] = filler_values[feature_index]
        return data

    @staticmethod
    def partition(data, labels, n):
        data, labels = DataHandler.shuffle(data, labels)

        first_data, first_labels = data[:n], labels[:n]
        rest_data, rest_labels = data[n:], labels[n:]
        return first_data, first_labels, rest_data, rest_labels

    @staticmethod
    def partition_data(d, n):
        td, tl = d['training_data'], d['training_labels']
        vd, vl, td, tl = DataHandler.partition(td, tl, n)
        d['training_data'], d['training_labels'] = td, tl
        d['validation_data'], d['validation_labels'] = vd, vl
        return d

    @staticmethod
    def shuffle(data, labels):
        indices = np.array(list(range(data.shape[0])))
        np.random.shuffle(indices)
        return data[indices], labels[indices]

    @staticmethod
    def oob(n, prob_dist=None):
        indices = np.arange(n)
        selected = np.random.choice(indices, size=n, p=prob_dist)
        indices[selected] = 0
        not_selected = np.nonzero(indices)[0]
        return selected, not_selected

    @staticmethod
    def sample_with_replacement(X, y):
        selected, not_selected = DataHandler.oob(X.shape[0])
        return X[selected], y[selected], X[not_selected], y[not_selected]

    @staticmethod
    def partitionIntoK(data, labels, k):
        indices = np.array(list(range(len(data))))
        np.random.shuffle(indices)

        factor = len(data) // k
        datasets = [data[indices[i * factor: (i + 1) * factor], :] for i in range(k)]
        labelsets = [labels[indices[i * factor: (i + 1) * factor], :] for i in range(k)]
        return datasets, labelsets

    @staticmethod
    def combine(datasets, labelsets, i):
        td = np.vstack([d for j, d in enumerate(datasets) if j != i])
        tl = np.vstack([d for j, d in enumerate(labelsets) if j != i])
        vd, vl = datasets[i], labelsets[i]
        return td, tl, vd, vl

    @staticmethod
    def results_to_csv(y_test, name='submission'):
        y_test = y_test.astype(int)
        df = pd.DataFrame({'Category': y_test})
        df.index += 1  # Ensures that the index starts at 1. 
        df.to_csv('%s.csv' % name, index_label='Id')

    @staticmethod
    def data_to_csv(data, name='data', fmt='%u'):
        if name.endswith('.csv'): name = name[:-4]
        np.savetxt('%s.csv' % name, data, delimiter=',', fmt=fmt)
