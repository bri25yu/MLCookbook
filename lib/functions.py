"""
@author bri25yu
Credit: EECS127 staff for function names, CS189 staff for gradient checking code!
"""

import numpy as np
from numpy.linalg import norm
from typing import Callable
from abc import abstractmethod

class Function():
    name = 'Function'

    @property
    @staticmethod
    @abstractmethod
    def initial_values():
        pass

    @property
    @staticmethod
    @abstractmethod
    def thresholds():
        pass

    @property
    @staticmethod
    @abstractmethod
    def coord_bounds():
        pass

    @property
    @staticmethod
    @abstractmethod
    def minimum():
        pass

    @property
    @staticmethod
    @abstractmethod
    def elev_azim():
        pass

    @staticmethod
    @abstractmethod
    def value(x1, x2):
        pass

    @staticmethod
    @abstractmethod
    def grad(x1, x2):
        pass

    @classmethod
    def check_fn_gradient(cls):
        X = np.random.randn(1, 2)

        # check the gradients w.r.t. each parameter
        print(
            "Relative error for %s function:" % cls.name,
            cls.check_gradients(
                fn=lambda x: cls.value(x[0][0], x[0][1]),  # the function we are checking
                grad=cls.grad(X[0][0], X[0][1]),  # the analytically computed gradient
                x=X,        # the variable w.r.t. which we are taking the gradient
            )
        )

    @staticmethod
    def check_gradients(
        fn: Callable[[np.ndarray], np.ndarray],
        grad: np.ndarray,
        x: np.ndarray,
        h: float = 1e-6,
    ) -> float:
        """Performs numerical gradient checking by numerically approximating
        the gradient using a two-sided finite difference.

        For each position in `x`, this function computes the numerical gradient as:
            numgrad = fn(x + h) - fn(x - h)
                    ---------------------
                                2h

        The function then returns the relative difference between the gradients:
            ||numgrad - grad||/||numgrad + grad||

        Parameters
        ----------
        fn       function whose gradients are being computed
        grad     supposed to be the gradient of `fn` at `x`
        x        point around which we want to calculate gradients
        h        a small number (used as described above)

        Returns
        -------
        relative difference between the numerical and analytical gradients
        """
        # ONLY WORKS WITH FLOAT VECTORS
        if x.dtype != np.float32 and x.dtype != np.float64:
            raise TypeError(f"`x` must be a float vector but was {x.dtype}")

        # initialize the numerical gradient variable
        numgrad = np.zeros_like(x)

        # compute the numerical gradient for each position in `x`
        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h
            pos = fn(x).copy()
            x[ix] = oldval - h
            neg = fn(x).copy()
            x[ix] = oldval

            # compute the derivative, also apply the chain rule
            numgrad[ix] = np.sum(pos - neg) / (2 * h)
            it.iternext()

        return norm(numgrad - grad) / norm(numgrad + grad)

class Booth(Function):
    name = 'booth'
    
    @staticmethod
    def initial_values():
        return np.array([8, 9])

    @staticmethod
    def thresholds():
        return 10**-7, 10**-14

    @staticmethod
    def coord_bounds():
        return -10, 10, 0.4

    @staticmethod
    def minimum():
        return np.array([1., 3.]).reshape(-1, 1)

    @staticmethod
    def elev_azim():
        return 30, -50

    @staticmethod
    def value(x1, x2):
        return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

    @staticmethod
    def grad(x1, x2):
        g1 = 10*x1 + 8*x2 - 34
        g2 = 8*x1 + 10*x2 - 38
        return np.stack((g1, g2), axis=-1)

class Beale(Function):
    name = 'beale'
    
    @staticmethod
    def initial_values():
        return np.array([3, 4])

    @staticmethod
    def thresholds():
        return 0.5, 0.07

    @staticmethod
    def coord_bounds():
        return -4.5, 4.5, 0.2

    @staticmethod
    def minimum():
        return np.array([3., .5]).reshape(-1, 1)

    @staticmethod
    def elev_azim():
        return 50, -140
        
    @staticmethod
    def value(x1, x2):
        return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

    @staticmethod
    def grad(x1, x2):
        in_1 = 1.5 + (x2 - 1) * x1
        in_2 = 2.25 + (x2 ** 2 - 1) * x1
        in_3 = 2.625 + (x2 ** 3 - 1) * x1
        g1 = 2*(x2 - 1)*in_1 + 2*(x2**2 - 1)*in_2 + 2*(x2**3 - 1)*in_3
        g2 = 2*x1*in_1 + 4*x1*x2*in_2 + 6*x1*(x2**2)*in_3
        return np.stack((g1, g2), axis=-1)

class Rosen2D(Function):
    name = 'rosen2d'
    
    @staticmethod
    def initial_values():
        return np.array([8, 9])

    @staticmethod
    def thresholds():
        return 10**-7, 10**-14

    @property
    @staticmethod
    def coord_bounds():
        return -5, 10, 0.3

    @staticmethod
    def minimum():
        return np.array([1., 1.]).reshape(-1, 1)

    @staticmethod
    def elev_azim():
        return 40, 140
        
    @staticmethod
    def value(x1, x2):
        return 100 * (x2 - x1**2)**2 + (x1 - 1)**2

    @staticmethod
    def grad(x1, x2):
        g1 = -400*x1*(x2 - x1**2) + 2*(x1-1)
        g2 = 200 * (x2 - x1**2)
        return np.stack((g1, g2), axis=-1)

class Ackley2D(Function):
    name = 'ackley2d'
    
    @staticmethod
    def initial_values():
        return np.array([25, 20])

    @staticmethod
    def thresholds():
        return 10**-12, 10**-12

    @staticmethod
    def coord_bounds():
        return -32.768, 32.768, 0.8192

    @staticmethod
    def minimum():
        return np.array([0., 0.]).reshape(-1, 1)

    @staticmethod
    def elev_azim():
        return 30, 40
        
    @staticmethod
    def value(x1, x2):
        return -20 * np.exp(-0.2 * np.sqrt((x1**2 + x2**2)/2)) - np.exp((np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))/2) + 20 + np.e

    @staticmethod
    def grad(x1, x2):
        in_11 = -20 * np.exp(-0.2 * np.sqrt((x1**2 + x2**2)/2))
        in_12 = -1 / (10 * np.sqrt(2) * np.sqrt(x1**2 + x2**2))
        in_1 = in_11 * in_12
        in_21 =  - np.exp((np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))/2)
        in_23 = 2 * np.pi
        in_2 = in_21 * in_23
        g1 = in_1 * 2 * x1 + in_2 * (-0.5 * np.sin(2*np.pi*x1))
        g2 = in_1 * 2 * x2 + in_2 * (-0.5 * np.sin(2*np.pi*x2))
        return np.stack((g1, g2), axis=-1)

FUNCTIONS = [Booth, Beale, Rosen2D, Ackley2D]
FUNCTIONS = {function.name : function for function in FUNCTIONS}

if __name__ == '__main__':
    for function in FUNCTIONS:
        function.check_fn_gradient()
