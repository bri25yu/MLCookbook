"""
@author bri25yu
"""

import numpy as np
from itertools import combinations
from collections import Counter

class VectorMath:
    
    @staticmethod
    def mean(data):
        return np.mean(data, axis=0) # assuming data is (n, d)

    @staticmethod
    def cov(data):
        return np.cov(data.T) # assuming data is (n, d)

    @staticmethod
    def normalize_vector(x):
        x -= np.mean(x)
        if var := np.var(x) != 0:
            x /= var
        return x

    @staticmethod
    def normalize(data):
        for i in range(data.shape[0]):
            VectorMath.normalize_vector(data[i])
        return data

    @staticmethod
    def add_lambda_I(cov, l):
        cov += l * np.eye(cov.shape[0])
        return cov

    @staticmethod
    def add_cov_matrices(covs):
        c = np.zeros(covs[0].shape)
        for cov in covs:
            c += cov
        return c

    @staticmethod
    def get_loss_by_class(expected, actual):
        c = Counter()
        for e, a in zip(expected, actual):
            if isinstance(e, np.ndarray): e = e[0]
            if isinstance(a, np.ndarray): a = a[0]
            if e != a: c[e] += 1
        return c

    @staticmethod
    def get_loss_func_for_class(c):
        def loss_func(expected, actual):
            loss = lambda i: (expected[i] == c or actual[i] == c) and expected[i] != actual[i]
            return sum(map(loss, range(expected.shape[0]))) / expected.shape[0]
        return loss_func

    @staticmethod
    def add_fictitious_dim(data):
        """
        Appends a column of ones as a fictitious dimension to the input data.

        Parameters
        ----------
        data : np.ndarray
            A (n, d)-shaped array of data.

        Returns
        -------
        data : np.ndarray
            A (n, d + 1)-shaped array of the exact same data, 
            except with a value of 1 appended to each data point.

        >>> vm = VectorMath()
        >>> data = np.reshape(np.array(list(range(6))), (3, 2))
        >>> expected = np.array([[0,1,1],[2,3,1],[4,5,1]])
        >>> np.all(vm.add_fictitious_dim(data).astype(int) == expected)
        True

        """
        return np.hstack((data, np.ones((data.shape[0], 1))))
    
    @staticmethod
    def get_loss(expected, actual):
        """
        Returns the error rate between the expected values and the actual values.
        Accepts any (n,) or (n,1) shaped values, as long as expected and actual share the same shape paradigm.

        Parameters
        ----------
        expected : np.ndarray
            The expected values.
        actual : np.ndarray
            The actual values.

        Returns
        -------
        loss : float
            The error rate as a percentage of values that are different, index-wise, between expected and actual.

        >>> vm = VectorMath()
        >>> expected = actual = np.array([1,2,3,4])
        >>> vm.get_loss(expected, actual)
        0.0
        >>> actual = np.array([1,2,3,3])
        >>> vm.get_loss(expected, actual)
        0.25
        >>> actual = np.array([1,2,3,float('inf')])
        >>> vm.get_loss(expected, actual)
        0.25
        >>> vm.get_loss(expected, -expected)
        1.0
        >>> expected = np.reshape(expected, (4, 1))
        >>> actual = np.reshape(np.array([1,2,3,4]), (4, 1))
        >>> vm.get_loss(expected, actual)
        0.0
        >>> actual = np.reshape(np.array([1,2,3,3]), (4, 1))
        >>> vm.get_loss(expected, actual)
        0.25
        >>> vm.get_loss(expected, -expected)
        1.0

        """
        return np.sum(expected != actual) / expected.shape[0]
    
    @staticmethod
    def entropy(y):
        c = np.bincount(y.ravel()) / y.shape[0]
        return -c @ np.nan_to_num(np.log(c))
    
    @staticmethod
    def gini_impurity(y):
        c = np.bincount(y.ravel()) / y.shape[0]
        return 1 - (c @ c)
    
    @staticmethod
    def get_split_fn(split_val, is_categorical):
        if is_categorical:
            return lambda v: np.isin(v, split_val)
        return lambda v: v <= split_val

    @staticmethod
    def split(x, split_fn):
        """
        Splits the indices of x into 2 groups based on x and split_fn:
            1) The indices corresponding to values in x where split_fn(val) is True.
            2) The indices corresponding to values in x where split_fn(val) is False.

        Parameters
        ----------
        x : np.ndarray
            An (n, 1)-shaped array of values.
        split_fn : func
            Boolean function to split on.

        Returns
        -------
        indices_less : np.ndarray
            The indices corresponding to values in x where split_fn(val) is True.
        indices_more : np.ndarray
            The indices corresponding to values in x where split_fn(val) is False.

        >>> vm = VectorMath()
        >>> x = 3 * np.array(range(10)) + 5
        >>> x = np.reshape(x, (x.shape[0], 1))
        >>> split_fn = lambda v: v <= 15
        >>> indices_less, indices_more = vm.split(x, split_fn)
        >>> indices_less.shape[0] + indices_more.shape[0] == x.shape[0]
        True
        >>> np.all(list(map(split_fn, x[indices_less])))
        True
        >>> not np.any(list(map(split_fn, x[indices_more])))
        True

        """
        values = split_fn(x)
        indices_less = np.where(values)[0]
        indices_more = np.where(np.logical_not(values))[0]
        return indices_less, indices_more
    
    @staticmethod
    def gain(x, y, split_fn, loss_func=None):
        """
        Calculates the gain using the input loss_func, G(split_fn).

        Parameters
        ----------
        x : np.ndarray
            An (n, 1)-shape array.
        y : np.ndarray
            An (n, 1)-shape array.
        split_fn : func
            Boolean function to split on.

        Returns
        -------
        gain : float
            The gain in splitting x on threshold. 

        """
        if not loss_func: loss_func = VectorMath.get_loss

        indices_less, indices_more = VectorMath.split(x, split_fn)
        y_less, y_more = y[indices_less], y[indices_more]

        base_loss = loss_func(y)
        less_loss = loss_func(y_less) * indices_less.shape[0]
        more_loss = loss_func(y_more) * indices_more.shape[0]
        return base_loss - ((less_loss + more_loss) / (x.shape[0]))
    
    @staticmethod
    def entropy_gain(x, y, split_fn):
        """
        Calculates the gain using the entropy function.
        """
        return VectorMath.gain(x, y, split_fn, VectorMath.entropy)
    
    @staticmethod
    def gini_gain(x, y, split_fn):
        """
        Calcualtes the gain using the gini impurity function.
        """
        return VectorMath.gain(x, y, split_fn, VectorMath.gini_impurity)

    @staticmethod
    def purity(x, y, split_fn, loss_func=None):
        if not loss_func: loss_func = VectorMath.get_loss

        indices_less, indices_more = VectorMath.split(x, split_fn)
        y_less, y_more = y[indices_less], y[indices_more]

        less_loss = loss_func(y_less) * indices_less.shape[0]
        more_loss = loss_func(y_more) * indices_more.shape[0]
        return -(less_loss + more_loss)
    
    @staticmethod
    def entropy_purity(x, y, split_fn):
        return VectorMath.purity(x, y, split_fn, VectorMath.entropy)
    
    @staticmethod
    def gini_purity(x, y, split_fn):
        return VectorMath.purity(x, y, split_fn, VectorMath.gini_impurity)

    @staticmethod
    def get_subsets(X, num_subsets):
        subsets = []
        for i in range(X.shape[1]):
            feature_subsets, vals = [], np.array(list(set(np.ravel(X[:, i]))))
            dim = vals.shape[0]

            if dim > 4:
                feature_subsets.extend(
                    np.random.choice(vals, size=int(np.random.random() * dim + 0.5), replace=False)
                        for _ in range(num_subsets))
            else:
                for s in range(1, dim):
                    feature_subsets.extend(vals[np.array(c)] for c in combinations(range(dim), s))
            subsets.append(feature_subsets)
        return subsets

    @staticmethod
    def get_thresholds(X, num_threshold_values, eps):
        minmax = zip(np.min(X, axis=0) + eps, np.max(X, axis=0) - eps)
        return np.array([np.linspace(l, h, num=num_threshold_values) for l, h in minmax])

if __name__ == '__main__':
    import doctest
    doctest.testmod()
