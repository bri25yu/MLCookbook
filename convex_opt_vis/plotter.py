from typing import Callable, Iterable, Literal, NamedTuple, Tuple, Union

from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.pyplot as plt


# Define a few numpy typing (npt) hints
class npt:
    NDArray = NDArray
    One = Literal[1]
    Two = Literal[2]
    Three = Literal[3]
    Len = int
    Shape = Tuple
    Matrix2D = NDArray[Shape[Two, Two]]
    Vector2D = NDArray[Shape[One, Two]]
    Scalar2D = NDArray[Shape[One, One]]
    NDArray1D = NDArray[Shape[Len]]
    MeshGrid = NDArray[Shape[Two, Len, Len]]
    Grid = NDArray[Shape[Len, Len]]
    MeshGrid2D = NDArray[Shape[Two, Len]]
    MeshGrid3D = NDArray[Shape[Three, Len]]


@dataclass
class Plotter3DArgs:
    x_1_min: float
    x_1_max: float
    x_2_min: float
    x_2_max: float
    n_partitions: int

    use_latex: bool


class PlotinequalityInfo(NamedTuple):
    Z: npt.Grid
    dX: npt.Grid = None
    dY: npt.Grid = None
    dZ: npt.Grid = None


class PlotinequalityInfoWithMask(NamedTuple):
    Z: npt.Grid
    dX: npt.Grid = None
    dY: npt.Grid = None
    dZ: npt.Grid = None
    mask: npt.Grid = None


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

    def plot_inequality(
        self, inequality: Callable[[npt.MeshGrid2D], npt.NDArray1D]
    ) -> None:
        if not self.is_initialized:
            self.initialize()

        plot_inequality_info = self.get_plot_inequality_info(inequality)
        self.plot_surface(*plot_inequality_info)

    def plot_intersection_of_inequalities(
        self, inequalities: Iterable[Callable[[npt.MeshGrid2D], npt.NDArray1D]]
    ) -> None:
        if not self.is_initialized:
            self.initialize()

        inequalities_info = list(map(self.get_plot_inequality_info, inequalities))

        # Augment inequalities info with masks
        create_mask = lambda f: np.zeros(f[0].shape, dtype=bool)
        augment_with_mask = lambda f: PlotinequalityInfoWithMask(*f, create_mask(f))
        inequalities_info = list(map(augment_with_mask, inequalities_info))

        # Get masks by intersection
        def mask_f1_with_f2(
            f1: PlotinequalityInfoWithMask, f2: PlotinequalityInfoWithMask
        ) -> PlotinequalityInfoWithMask:
            f1_mask = f1.mask
            f1_Z = f1.Z
            f2_Z = f2.Z
            f2_dZ = f2.dZ

            compare_op = np.less if f2_dZ[0][0] > 0 else np.greater

            f1_compared_to_f2 = compare_op(f1_Z, f2_Z)
            f1_mask = f1_mask | f1_compared_to_f2

            new_f1 = PlotinequalityInfoWithMask(*f1[:-1], f1_mask)

            return new_f1

        for f1_idx in range(len(inequalities_info)):
            f1 = inequalities_info[f1_idx]
            for f2_idx in range(len(inequalities_info)):
                if f1_idx == f2_idx: continue

                f2 = inequalities_info[f2_idx]

                f1 = mask_f1_with_f2(f1, f2)

            inequalities_info[f1_idx] = f1

        # Apply masks
        apply_mask = lambda f: PlotinequalityInfo(*self.apply_mask(f[:-1], f.mask))
        inequalities_info = list(map(apply_mask, inequalities_info))

        # Plot inequalities
        list(map(self.plot_surface, inequalities_info))

    def plot_surface(
        self, plot_inequality_info: PlotinequalityInfo
    ) -> None:
        ax = self.ax
        meshgrid = self.meshgrid

        Z, dX, dY, dZ = plot_inequality_info

        # Plot inequality
        X, Y = meshgrid
        ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, edgecolor="none"
        )

        # Plot normal vectors
        not_none = lambda o: o is not None
        if all(map(not_none, (dX, dY, dZ))):
            quiver_args = (X, Y, Z, dX, dY, dZ)
            ax.quiver(*self.apply_downsample_mask(*quiver_args))

    def apply_downsample_mask(self, *args: Iterable[npt.Grid]) -> Tuple[npt.Grid]:
        """
        All args must be the same shape
        """
        shape = args[0].shape

        mask = np.ones(shape)
        mask[::10, ::10] = 0

        return self.apply_mask(args, mask)

    def apply_mask(self, grids: Iterable[npt.Grid], mask: npt.Grid) -> Tuple[npt.Grid]:
        mask_grid = lambda grid: np.ma.masked_where(mask, grid)
        return tuple(map(mask_grid, grids))

    def get_plot_inequality_info(
        self, f: Callable[[npt.MeshGrid2D], npt.NDArray1D]
    ) -> PlotinequalityInfo:
        """
        Returns a tuple of (Z, dX, dY, dZ)
        """
        meshgrid = self.meshgrid

        original_shape = meshgrid[0].shape
        meshgrid2d: npt.MeshGrid2D = np.reshape(meshgrid, (2, -1))

        # inequality
        Z = f(meshgrid2d)
        Z = np.reshape(Z, original_shape)

        # Normal vectors
        normal_vectors = f.df(meshgrid2d)
        normal_vectors = normal_vectors.reshape((3, *original_shape))

        return PlotinequalityInfo(Z, *normal_vectors)

    def get_ax(self) -> Union[Axes, None]:
        return self.ax
