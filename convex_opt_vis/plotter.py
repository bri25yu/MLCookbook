from typing import Callable, Literal, Tuple, Union

from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.pyplot as plt


# Define a few numpy typing (npt) hints
class npt:
    NDArray = NDArray
    Two = Literal[2]
    Len = int
    Shape = Tuple
    NDArray1D = NDArray[Shape[Len]]
    MeshGrid = NDArray[Shape[Two, Len, Len]]
    MeshGrid2D = NDArray[Shape[Two, Len]]


@dataclass
class Plotter3DArgs:
    x_1_min: float
    x_1_max: float
    x_2_min: float
    x_2_max: float
    n_partitions: int

    use_latex: bool


class Plotter3D:
    def __init__(self, args: Plotter3DArgs):
        # Setup LaTeX if necessary
        use_latex = args.use_latex
        if use_latex:
            plt.rcParams["text.usetex"] = True

        # Setup our plotting meshgrid
        # First, we retrieve relevant properties from our args
        x_1_min = args.x_1_min
        x_1_max = args.x_1_max
        x_2_min = args.x_2_min
        x_2_max = args.x_2_max
        n_partitions = args.n_partitions

        ranges = [
            (x_1_min, x_1_max, n_partitions),
            (x_2_min, x_2_max, n_partitions),
        ]

        # We get a grid of points
        def create_axis_points(axis_min: float, axis_max: float, n_partitions: int) -> npt.NDArray1D:
            return np.linspace(axis_min, axis_max, n_partitions)

        points = [create_axis_points(*r) for r in ranges]

        # Create a meshgrid using our x and y points
        meshgrid: npt.MeshGrid = np.array(np.meshgrid(*points))

        # Store our input plotter args and meshgrid
        self.args = args
        self.meshgrid: npt.MeshGrid = meshgrid

        # Setup our plotting vars
        self.fig: Figure = None
        self.ax: Axes = None

    @property
    def is_initialized(self) -> bool:
        return self.fig is not None and self.ax is not None

    def initialize(self):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.set_zlabel("f(x_1, x_2)")

        self.fig = fig
        self.ax = ax

    def show(self) -> None:
        if not self.is_initialized:
            return

        fig = self.fig

        fig.tight_layout()
        plt.show()

    def plot_function(
        self, f: Callable[[npt.MeshGrid2D], npt.NDArray1D]
    ) -> None:
        if not self.is_initialized:
            self.initialize()

        meshgrid = self.meshgrid
        ax = self.ax

        meshgrid2d: npt.MeshGrid2D = np.reshape(meshgrid, (2, -1))

        X = meshgrid[0]
        Y = meshgrid[1]
        Z = f(meshgrid2d)

        Z = np.reshape(Z, (X.shape[0], X.shape[1]))

        ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, edgecolor="none"
        )

    def get_ax(self) -> Union[Axes, None]:
        return self.ax
