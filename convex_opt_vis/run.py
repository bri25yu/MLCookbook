from typing import List

import sys

sys.path += ":."

from argparse import ArgumentParser

import numpy as np

from convex_opt_vis.plotter import Plotter3D, Plotter3DArgs
from convex_opt_vis.inequalities import (
    AbstractInequality, AffineInequality, QuadraticInequality
)


def get_inequalities(shape: str) -> List[AbstractInequality]:
    return INEQUALITY_OPTIONS[shape]


def main(shape: str):
    plotter = initialize_plotter()
    inequalities = get_inequalities(shape)

    plotter.plot_intersection_of_inequalities(inequalities)

    plotter.show()


def initialize_plotter() -> Plotter3D:
    args = Plotter3DArgs(
        -6, 6, -6, 6, 50, False
    )
    plotter = Plotter3D(args)
    plotter.initialize()

    return plotter


INEQUALITY_OPTIONS = {
    "hexagon": [
        AffineInequality(np.array([[0, 0]]), np.array([[1]]), -1),
        AffineInequality(np.array([[0, 0]]), np.array([[-1]]), 1),
        AffineInequality(np.array([[0, 1]]), np.array([[-2]]), 1),
        AffineInequality(np.array([[0, -1]]), np.array([[-2]]), 1),
        AffineInequality(np.array([[0, 1]]), np.array([[2]]), -1),
        AffineInequality(np.array([[0, -1]]), np.array([[2]]), -1),
    ],
    "paraboloid": [
        AffineInequality(np.array([[0, 0]]), np.array([[10]]), -1),
        QuadraticInequality(np.array([[1, 0], [0, 1]]), np.array([[0, 0]]), np.array([[0]]), 1),
    ],
    "ellipsoid": [
        AffineInequality(np.array([[0, 0]]), np.array([[10]]), -1),
        QuadraticInequality(np.array([[1, 1], [0, 1]]), np.array([[0, 0]]), np.array([[0]]), 1),
    ],
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--shape", "-s", choices=INEQUALITY_OPTIONS
    )
    args = parser.parse_args()

    main(args.shape)
