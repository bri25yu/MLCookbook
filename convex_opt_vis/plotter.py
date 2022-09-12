from typing import Iterable, NamedTuple, Tuple, Union

from functools import reduce

from dataclasses import dataclass

import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.pyplot as plt

from convex_opt_vis.typing import npt
from convex_opt_vis.inequalities import AbstractInequality


@dataclass
class Plotter3DArgs:
    cube_halfwidth: float
    n_partitions: int
    downsample_mask_interval: int = 5


class PlotPolygonInfo(NamedTuple):
    edge_mask: npt.Grid
    interior_mask: npt.Grid


class Plotter3D:
    def __init__(self, args: Plotter3DArgs):
        # Setup our plotting meshgrid
        cube_halfwidth = args.cube_halfwidth
        n_partitions = args.n_partitions
        start = -cube_halfwidth
        end = cube_halfwidth

        create_axis_points = lambda: np.linspace(start, end, n_partitions)
        points = [create_axis_points() for _ in range(3)]

        # Create a meshgrid using our x and y points
        meshgrid: npt.MeshGrid = np.array(np.meshgrid(*points))

        # Store our input plotter args and meshgrid
        self.args = args
        self.atol = 2 * cube_halfwidth / n_partitions
        self.meshgrid: npt.MeshGrid = meshgrid

        # Setup our plotting vars
        self.fig: Figure = None
        self.ax: Axes = None

    # -------------------------------------------
    # START plotter obj-related methods
    # -------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self.fig is not None and self.ax is not None

    def initialize(self):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        self.fig = fig
        self.ax = ax

    def show(self) -> None:
        if not self.is_initialized:
            return

        fig = self.fig

        fig.tight_layout()
        plt.show()

    def get_ax(self) -> Union[Axes, None]:
        return self.ax

    # -------------------------------------------
    # START exposed plotting methods
    # -------------------------------------------

    def plot_inequality(self, inequality: AbstractInequality) -> None:
        if not self.is_initialized:
            self.initialize()

        plot_inequality_info = self.get_plot_inequality_info(inequality)
        self.plot_polygon(plot_inequality_info)

    def plot_intersection_of_inequalities(self, inequalities: Iterable[AbstractInequality]) -> None:
        if not self.is_initialized:
            self.initialize()

        inequalities_info = list(map(self.get_plot_inequality_info, inequalities))

        # Get plotting information
        interior_mask = reduce(np.logical_or, map(lambda f: f.interior_mask, inequalities_info))
        surface_masks = list(map(lambda f: f.edge_mask | interior_mask, inequalities_info))

        # Plot
        self.plot_polygon_interior(interior_mask)
        list(map(self.plot_polygon_surface, surface_masks))

    # -------------------------------------------
    # START plot_polygon related methods
    # -------------------------------------------

    def plot_polygon(self, plot_polygon_info: PlotPolygonInfo) -> None:
        edge_mask, interior_mask = plot_polygon_info

        self.plot_polygon_surface(edge_mask)
        self.plot_polygon_interior(interior_mask)

    def get_plot_inequality_info(self, f: AbstractInequality) -> PlotPolygonInfo:
        meshgrid = self.meshgrid
        atol = self.atol

        original_shape = meshgrid[0].shape
        meshgrid3d: npt.MeshGrid3D = np.reshape(meshgrid, (3, -1))

        output = f(meshgrid3d)

        # Equality case for surface edges
        offset = np.ones_like(output) * f.offset
        edge_mask = ~np.isclose(output, offset, atol=atol)

        # Inequality case for interior points
        # Invert the inequality for mask
        interior_mask = ~(output <= f.offset)

        # Reshape both masks to original shape
        edge_mask = np.reshape(edge_mask, original_shape)
        interior_mask = np.reshape(interior_mask, original_shape)

        return PlotPolygonInfo(edge_mask, interior_mask)

    def plot_polygon_surface(self, edge_mask: npt.Grid) -> None:
        ax = self.ax
        meshgrid = self.meshgrid

        X, Y, Z = meshgrid

        # Reshape 3d meshgrid to 2d to be usable by axes plotting
        X_2d = X[0, :, :].T
        Y_2d = Y[:, 0, :]

        # Plot surface points using equality/edge mask
        Z_surface_3d = self.apply_mask((Z,), edge_mask)[0]
        Z_surface_2d = Z_surface_3d.mean(axis=2).round(2)
        ax.plot_surface(
            X_2d, Y_2d, Z_surface_2d, rstride=1, cstride=1, edgecolor="none"
        )

    def plot_polygon_interior(self, interior_mask: npt.Grid) -> None:
        ax = self.ax
        meshgrid = self.meshgrid
        downsample_mask = self.downsample_mask

        # Plot interior points using inequality/interior mask
        interior_mask_total = interior_mask | downsample_mask
        meshgrid_interior = self.apply_mask(meshgrid, interior_mask_total)
        ax.scatter(*meshgrid_interior)

    # -------------------------------------------
    # START masking helpers
    # -------------------------------------------

    @property
    def downsample_mask(self) -> npt.Grid:
        meshgrid = self.meshgrid
        downsample_mask_interval = self.args.downsample_mask_interval

        shape = meshgrid[0].shape

        slice_interval = slice(None, None, downsample_mask_interval)
        mask_slices = [slice_interval] * len(shape)

        mask = np.ones(shape, dtype=bool)
        mask[mask_slices] = 0

        return mask

    def apply_mask(self, grids: Iterable[npt.Grid], mask: npt.Grid) -> Tuple[npt.Grid]:
        mask_grid = lambda grid: np.ma.masked_where(mask, grid)
        return tuple(map(mask_grid, grids))
