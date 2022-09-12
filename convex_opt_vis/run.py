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
    args = Plotter3DArgs(6, 50, 5)
    plotter = Plotter3D(args)
    plotter.initialize()

    return plotter


INEQUALITY_OPTIONS = {
    "plane": [
        AffineInequality(1, np.array([[0, 0, 1]])),
    ],
    "hexagon": [
        AffineInequality(1, np.array([[0, 0, 1]])),
        AffineInequality(1, np.array([[0, 0, -1]])),
        AffineInequality(3, np.array([[0, 1, 1]])),
        AffineInequality(3, np.array([[0, 1, -1]])),
        AffineInequality(3, np.array([[0, -1, 1]])),
        AffineInequality(3, np.array([[0, -1, -1]])),
    ],
    "ellipsoid": [
        QuadraticInequality(9, np.eye(3), np.zeros((3, 1)), np.array([[0]])),
    ],
    "ellipsoids": [
        QuadraticInequality(
            4, np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros((3, 1)), np.array([[0]])
        ),
        QuadraticInequality(
            4, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]]), np.zeros((3, 1)), np.array([[0]])
        ),
        QuadraticInequality(
            4, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]]), np.zeros((3, 1)), np.array([[0]])
        ),
    ],
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--shape", "-s", choices=INEQUALITY_OPTIONS, required=True
    )
    args = parser.parse_args()

    main(args.shape)
