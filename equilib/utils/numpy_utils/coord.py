#!/usr/bin/env python3

import numpy as np


def create_coord(
    height: int,
    width: int,
) -> np.ndarray:
    """Create mesh coordinate grid with height and width

    `z-axis` scale is `1`

    params:
    - height (int)
    - width (int)

    return:
    - coordinate (np.ndarray)
    """
    _xs = np.linspace(0, width - 1, width)
    _ys = np.linspace(0, height - 1, height)
    xs, ys = np.meshgrid(_xs, _ys)
    zs = np.ones_like(xs)
    coord = np.stack((xs, ys, zs), axis=2)
    return coord
