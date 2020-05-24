"""
@author bri25yu
"""

import numpy as np

from lib.functions import FUNCTIONS
from lib.visualize import Optimization
from lib.algorithms import HyperparameterTuning
from optimization.descent import DESCENT_METHODS
from optimization.settings import OUTPUT_DIR

booth_params = dict(
    method='adagrad',
    lr=lambda itr: 10,
    num_iters=100
)

beale_params = dict(
    method='adagrad',
    lr=lambda itr: 5 if itr < 10 else 1980,
    num_iters=10**2
)

rosen2d_params = dict(
    method='adagrad',
    lr=lambda itr: 7.8738327606771,
    num_iters=2000,
)

ackley2d_params = dict(
    method='momentum',
    lr=lambda itr: 5,
    num_iters=15,
    alpha=0.8
)
    