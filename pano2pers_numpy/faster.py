#!/usr/bin/env python3

from typing import List

import numpy as np


def linear_interp(v0, v1, d, l):
    r"""Basic Linear Interpolation
    """
    return v0*(1-d)/l + v1*d/l


def interp2d(
    Q: List[np.ndarray],
    dy: np.array, dx: np.array,
    mode: str = 'bilinear',
) -> np.ndarray:
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
        raise NotImplementedError
    else:
        raise NotImplementedError


def grid_sample(
    img: np.array, grid: np.array,
    mode: str = 'bilinear',
) -> np.array:
    r"""Numpy Grid Sample
    """
    channels, h_in, w_in = img.shape
    _, h_out, w_out = grid.shape

    # Image conversion values (Lookup Image dtype)
    if img.dtype == np.uint8:
        # uint8 is faster
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
    #NOTE: uint8 convertion causes truncation, so use uint64
    max_grid = min_grid + 1
    d_grid = grid - min_grid

    max_grid[0,:,:] = np.where(
        max_grid[0,:,:] >= h_in,
        max_grid[0,:,:] - h_in,
        max_grid[0,:,:])
    max_grid[1,:,:] = np.where(
        max_grid[1,:,:] >= w_in,
        max_grid[1,:,:] - w_in,
        max_grid[1,:,:])

    y_mins = min_grid[0,:,:]
    x_mins = min_grid[1,:,:]
    y_mins = y_mins.flatten()
    x_mins = x_mins.flatten()

    y_maxs = max_grid[0,:,:]
    x_maxs = max_grid[1,:,:]
    y_maxs = y_maxs.flatten()
    x_maxs = x_maxs.flatten()

    y_d = d_grid[0,:,:]
    x_d = d_grid[1,:,:]
    y_d = y_d.flatten()
    x_d = x_d.flatten()

    Q00 = img[:,y_mins,x_mins]
    Q10 = img[:,y_maxs,x_mins]
    Q01 = img[:,y_mins,x_maxs]
    Q11 = img[:,y_maxs,x_maxs]

    out = interp2d(
        [Q00, Q10, Q01, Q11],
        y_d, x_d,
        mode='bilinear')

    out = np.where(out >= _max, _max, out)
    out = np.where(out < _min, _min, out)
    out = out.reshape(channels, h_out, w_out)
    return out.astype(_dtype)