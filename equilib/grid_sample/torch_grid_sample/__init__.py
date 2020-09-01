#!/usr/bin/env python3

from .basic import grid_sample as custom
from .torch_func import grid_sample as torch

__all__ = [
    "torch",
    "custom",
]
