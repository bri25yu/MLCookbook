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


class AbstractInequality(AbstractFunction):
    @abstractmethod
    def df(self, x: npt.MeshGrid2D) -> npt.MeshGrid3D:
        pass


class AffineInequality(AbstractInequality):
    def __init__(self, a: npt.NDArray1D, b: npt.NDArray1D, direction: float) -> None:
        self.a = a
        self.b = b
        # -1 for a^Tx + b \leq 1 or 1 for a^Tx + b \geq 1
        self.direction = direction

    def f(self, x: npt.MeshGrid2D) -> npt.NDArray1D:
        a = self.a
        b = self.b

        return a @ x + b

    def df(self, x: npt.MeshGrid2D) -> npt.MeshGrid3D:
        a = self.a
        b = self.b

        n = x.shape[1]

        # Calculate gradients
        xy_grad = a.T
        z_grad = -1 * np.ones(b.shape).T
        grad = - self.direction * np.concatenate((xy_grad, z_grad))

        # Reshape to a downstream usable form
        grad_reshaped = np.tile(grad, (1, n))

        return grad_reshaped

    @classmethod
    def create_random(cls):
        a: npt.NDArray1D = 2 * np.random.rand(1, 2) - 1
        b: npt.NDArray1D = 2 * np.random.rand(1, 1) - 1
        direction: float = 2 * np.random.randint(0, 2) - 1

        return cls(a, b, direction)
