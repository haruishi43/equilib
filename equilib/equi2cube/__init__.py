#!/usr/bin/env python3

from .equi2cube_numpy import Equi2Cube as NumpyEqui2Cube
from .equi2cube_torch import Equi2Cube as TorchEqui2Cube

__all__ = [
    "NumpyEqui2Cube",
    "TorchEqui2Cube",
]
