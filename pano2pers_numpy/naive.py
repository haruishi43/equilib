#!/usr/bin/env python3

from typing import List

import numpy as np


def linear_interp(v0, v1, d, l):
    return v0*(1-d)/l + v1*d/l


def interp2d(
    Q: List[np.array],
    dy: float,
    dx: float,
    mode: str = 'bilinear',
) -> np.array:
    r"""Naive Interpolation
        (y,x): target pixel
        mode: interpolation mode
    """
    q00, q10, q01, q11 = Q
    if mode == 'bilinear':
        f0 = linear_interp(q00, q01, dx, 1)
        f1 = linear_interp(q10, q11, dx, 1)
        return linear_interp(f0, f1, dy, 1)
    elif mode == 'nearest':
        if dx < 0.5:
            if dy < 0.5:
                return q00
            else:
                return q10
        else:
            if dy < 0.5:
                return q01
            else:
                return q11
    else:
        print(f"{mode} is not supported")
        return None


def grid_sample(
    img: np.array,
    grid: np.array,
    mode: str = 'bilinear',
) -> np.array:
    r"""Naive grid sample algorithm
    """
    channels, h_in, w_in = img.shape
    _, h_out, w_out = grid.shape

    # Image conversion values
    if img.dtype == np.uint8:
        _min = 0
        _max = 255
        _dtype = np.uint8
    elif img.dtype == np.float64:
        _min = 0.0
        _max = 1.0
        _dtype = np.float64
    else:
        print(f"{img.dtype} is not supported")

    # Initialize output image
    out = np.zeros((channels, h_out, w_out), dtype=_dtype)

    min_grid = np.floor(grid).astype(np.uint64)
    # NOTE: uint8 convertion causes truncation, so use uint64
    max_grid = min_grid + 1
    d_grid = grid - min_grid

    max_grid[0, :, :] = np.where(
        max_grid[0, :, :] >= h_in,
        max_grid[0, :, :] - h_in,
        max_grid[0, :, :]
    )
    max_grid[1, :, :] = np.where(
        max_grid[1, :, :] >= w_in,
        max_grid[1, :, :] - w_in,
        max_grid[1, :, :]
    )

    for y in range(h_out):
        for x in range(w_out):
            # _y, _x = grid[:,y,x]
            dy, dx = d_grid[:, y, x]
            y0, x0 = min_grid[:, y, x]
            y1, x1 = max_grid[:, y, x]

            q00 = img[:, y0, x0]
            q10 = img[:, y1, x0]
            q01 = img[:, y0, x1]
            q11 = img[:, y1, x1]

            out[:, y, x] = interp2d([q00, q10, q01, q11], dy, dx, mode=mode)

    out = np.where(out >= _max, _max, out)
    out = np.where(out < _min, _min, out)
    return out.astype(_dtype)
