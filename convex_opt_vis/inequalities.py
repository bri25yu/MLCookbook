from abc import abstractclassmethod, abstractmethod

import numpy as np

from convex_opt_vis.typing import npt


class AbstractFunction:
    def __call__(self, x: npt.MeshGrid3D) -> npt.NDArray1D:
        return self.f(x)

    @abstractmethod
    def f(self, x: npt.MeshGrid3D) -> npt.NDArray1D:
        pass

    @abstractclassmethod
    def create_random(cls):
        pass


class AbstractInequality(AbstractFunction):
    """
    f(x) <= offset
    """
    def __init__(self, offset: float) -> None:
        self.offset = offset


class AffineInequality(AbstractInequality):
    """
    f(x) = a^Tx
    """

    def __init__(self, offset: float, a: npt.Vector3D) -> None:
        super().__init__(offset)

        self.a = a.reshape(3, 1)

    def f(self, x: npt.MeshGrid3D) -> npt.NDArray1D:
        a = self.a

        return a.T @ x

    @classmethod
    def create_random(cls):
        a: npt.NDArray1D = 2 * np.random.rand(3, 1) - 1
        offset: float = 2 * np.random.rand() - 1

        return cls(offset, a)


class QuadraticInequality(AbstractInequality):
    """
    f(x) = 1/2 x^Tpx + q^Tx + r
    """

    def __init__(
        self, offset: float, p: npt.Matrix3D, q: npt.Vector3D, r: npt.Scalar3D
    ) -> None:
        super().__init__(offset)

        self.p = p
        self.q = q
        self.r = r

    def f(self, x: npt.MeshGrid3D) -> npt.NDArray1D:
        p = self.p
        q = self.q
        r = self.r

        return 0.5 * np.sum((p @ x) * x, axis=0) + q.T @ x + r

    @classmethod
    def create_random(cls):
        pass
