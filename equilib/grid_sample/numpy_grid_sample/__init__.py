#!/usr/bin/env python

from .naive import grid_sample as naive
from .faster import grid_sample as faster

__all__ = [
    "faster",
    "naive",
]
