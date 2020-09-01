#!/usr/bin/env python3

from . import numpy_grid_sample as numpy_func
from . import torch_grid_sample as torch_func

__all__ = [
    "numpy_func",
    "torch_func",
]
