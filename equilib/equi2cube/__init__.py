#!/usr/bin/env python3

from .equi2cube_numpy.equi2cube import Equi2Cube as NumpyEqui2Cube
from .equi2cube_torch.equi2cube import Equi2Cube as TorchEqui2Cube

__all__ = [
    "NumpyEqui2Cube",
    "TorchEqui2Cube",
]
