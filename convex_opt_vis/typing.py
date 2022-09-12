from typing import Literal, Tuple
from numpy.typing import NDArray


# Define a few numpy typing (npt) hints
class npt:
    NDArray = NDArray
    One = Literal[1]
    Three = Literal[3]
    Len = int
    Shape = Tuple
    Matrix3D = NDArray[Shape[Three, Three]]
    Vector3D = NDArray[Shape[Three, One]]
    Scalar3D = NDArray[Shape[One]]
    NDArray1D = NDArray[Shape[Len]]
    MeshGrid = NDArray[Shape[Three, Len, Len, Len]]
    Grid = NDArray[Shape[Len, Len, Len]]
    MeshGrid3D = NDArray[Shape[Three, Len]]
