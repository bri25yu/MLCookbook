from typing import List

import sys

sys.path += ":."

import numpy as np

from convex_opt_vis.plotter import Plotter3D, Plotter3DArgs
from convex_opt_vis.inequalities import AbstractInequality, AffineInequality


def get_inequalities() -> List[AbstractInequality]:
    return [
        AffineInequality(np.array([[0, 0]]), np.array([[1]]), -1),
        AffineInequality(np.array([[0, 0]]), np.array([[-1]]), 1),
        AffineInequality(np.array([[0, 1]]), np.array([[-2]]), 1),
        AffineInequality(np.array([[0, -1]]), np.array([[-2]]), 1),
        AffineInequality(np.array([[0, 1]]), np.array([[2]]), -1),
        AffineInequality(np.array([[0, -1]]), np.array([[2]]), -1),
    ]


def main():
    plotter = initialize_plotter()
    inequalities = get_inequalities()

    plotter.plot_intersection_of_inequalities(inequalities)

    plotter.show()


def initialize_plotter() -> Plotter3D:
    args = Plotter3DArgs(
        -6, 6, -6, 6, 50, False
    )
    plotter = Plotter3D(args)
    plotter.initialize()

    return plotter


if __name__ == "__main__":
    main()
