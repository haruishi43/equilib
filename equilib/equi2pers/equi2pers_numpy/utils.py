#!/usr/bin/env python3

from typing import Tuple

import numpy as np


def create_coord(
    height: int,
    width: int,
) -> np.ndarray:
    r"""Create mesh coordinate grid with height and width

    `z-axis` scale is `1`

    params:
        height: int
        width: int

    return:
        coordinate: numpy.ndarray
    """
    _xs = np.linspace(0, width - 1, width)
    _ys = np.linspace(0, height - 1, height)
    xs, ys = np.meshgrid(_xs, _ys)
    zs = np.ones_like(xs)
    coord = np.stack((xs, ys, zs), axis=2)
    return coord


def create_K(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
) -> np.ndarray:
    r"""Create Intrinsic Matrix

    params:
        height: int
        width: int
        fov_x: float
        skew: float

    return:
        K: 3x3 matrix numpy.ndarray

    NOTE:
        ref: http://ksimek.github.io/2013/08/13/intrinsic/
    """
    # perspective projection (focal length)
    f = width / (2.0 * np.tan(np.radians(fov_x) / 2.0))
    # transform between camera frame and pixel coordinates
    K = np.array([[f, skew, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])
    return K


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


def create_rot_mat(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    r"""Create Rotation Matrix

    params:
        roll: x-axis rotation float
        pitch: y-axis rotation float
        yaw: z-axis rotation float

    return:
        rotation matrix: numpy.ndarray

    Camera coordinates -> z-axis points forward, y-axis points upward
    Global coordinates -> x-axis points forward, z-axis poitns upward
    """

    # default rotation that changes global to camera coordinates
    x = np.pi
    y = np.pi
    z = np.pi
    R = create_rotation_matrix(x=x, y=y, z=z)

    # rotation matrix
    R = R @ create_rotation_matrix(x=roll, y=pitch, z=yaw)
    return R


def pixel_wise_rot(M: np.ndarray) -> Tuple[np.ndarray]:
    r"""Rotation coordinates to phi/theta of the equirectangular image

    params:
        M: np.ndarray

    return:
        phis: np.ndarray
        thetas: np.ndarray
    """
    phis = np.arcsin(M[:, :, 1] / np.linalg.norm(M, axis=-1))
    thetas = np.arctan2(M[:, :, 0], M[:, :, 2])
    return phis, thetas
