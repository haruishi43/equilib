#!/usr/bin/env python

from .faster import grid_sample as faster
from .naive import grid_sample as naive

__all__ = [
    'faster',
    'naive',
]
