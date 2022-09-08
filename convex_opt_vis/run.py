import sys

sys.path += ":."

import numpy as np

from convex_opt_vis.plotter import Plotter3D, Plotter3DArgs, npt


args = Plotter3DArgs(
    -6, 6, -6, 6, 30, False
)
plotter = Plotter3D(args)
plotter.initialize()

A: npt.NDArray1D = np.random.rand(1, 2)
b: npt.NDArray1D = np.random.rand(1, 1)
def f(x: npt.MeshGrid2D) -> npt.NDArray1D:
    print(A.shape, x.shape, b.shape)
    return A @ x + b


plotter.plot_function(f)
plotter.show()

