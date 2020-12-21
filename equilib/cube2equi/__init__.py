#!/usr/bin/env python3

from .cube2equi_numpy import Cube2Equi as NumpyCube2Equi
from .cube2equi_torch import Cube2Equi as TorchCube2Equi

__all__ = [
    "NumpyCube2Equi",
    "TorchCube2Equi",
]
