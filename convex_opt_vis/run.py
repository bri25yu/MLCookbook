import sys

sys.path += ":."

from convex_opt_vis.plotter import Plotter3D, Plotter3DArgs
from convex_opt_vis.functions import AffineFunction


args = Plotter3DArgs(
    -6, 6, -6, 6, 30, False
)
plotter = Plotter3D(args)
plotter.initialize()

function1 = AffineFunction.create_random()
function2 = AffineFunction.create_random()

plotter.plot_function(function1)
plotter.plot_function(function2)
plotter.show()

