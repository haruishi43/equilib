#!/usr/bin/env python3

import numpy as np


def create_rotation_matrix(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    r"""Create Rotation Matrix

    params:
        x: x-axis rotation float
        y: y-axis rotation float
        z: z-axis rotation float

    return:
        rotation matrix: numpy.ndarray
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


def cube_h2list(cube_h):
    assert cube_h.shape[-2] * 6 == cube_h.shape[-1]
    return np.split(cube_h, 6, axis=-1)


def cube_h2dict(cube_h):
    cube_list = cube_h2list(cube_h)
    return dict(
        [
            (k, cube_list[i])
            for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
        ]
    )


def cube_h2dice(cube_h):
    assert cube_h.shape[-2] * 6 == cube_h.shape[-1]
    w = cube_h.shape[-2]
    cube_dice = np.zeros((cube_h.shape[0], w * 3, w * 4), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        cube_dice[:, sy * w : (sy + 1) * w, sx * w : (sx + 1) * w] = cube_list[
            i
        ]
    return cube_dice
