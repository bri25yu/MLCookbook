from abc import abstractclassmethod, abstractmethod

import numpy as np

from convex_opt_vis.plotter import npt


class AbstractFunction:
    def __call__(self, x: npt.MeshGrid2D) -> npt.NDArray1D:
        return self.f(x)

    @abstractmethod
    def f(self, x: npt.MeshGrid2D) -> npt.NDArray1D:
        pass

    @abstractclassmethod
    def create_random(cls):
        pass


class AffineFunction(AbstractFunction):
    def __init__(self, a: npt.NDArray1D, b: npt.NDArray1D) -> None:
        self.a = a
        self.b = b

    def f(self, x: npt.MeshGrid2D) -> npt.NDArray1D:
        a = self.a
        b = self.b

        return a @ x + b

    def df(self, x: npt.MeshGrid2D) -> npt.MeshGrid2D:
        a = self.a

        n = x.shape[1]

        return np.tile(a, (n, 1)).T

    @classmethod
    def create_random(cls):
        a: npt.NDArray1D = np.random.rand(1, 2)
        b: npt.NDArray1D = np.random.rand(1, 1)

        return cls(a, b)
