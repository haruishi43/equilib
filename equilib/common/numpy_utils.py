#!/usr/bin/env python3

import numpy as np


def create_coord(
    height: int,
    width: int,
) -> np.ndarray:
    r"""Create mesh coordinate grid with height and width

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


def create_rotation_matrix(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    r"""Create Rotation Matrix

    params:
    - x: x-axis rotation float
    - y: y-axis rotation float
    - z: z-axis rotation float

    return:
    - rotation matrix (np.ndarray)
    """
    # calculate rotation about the x-axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x), -np.sin(x)],
            [0.0, np.sin(x), np.cos(x)],
        ]
    )
    # calculate rotation about the y-axis
    R_y = np.array(
        [
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)],
        ]
    )
    # calculate rotation about the z-axis
    R_z = np.array(
        [
            [np.cos(z), -np.sin(z), 0.0],
            [np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return R_z @ R_y @ R_x
