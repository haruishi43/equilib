#!/usr/bin/env python3

import numpy as np


def create_global2camera_rotation_matrix() -> np.ndarray:
    """Rotation from global (world) to camera coordinates"""
    R_XY = np.array(
        [  # X <-> Y
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R_YZ = np.array(
        [  # Y <-> Z
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return R_XY @ R_YZ


def create_intrinsic_matrix(
    w_pers: int,
    h_pers: int,
    fov_x: float,
    skew: float,
) -> np.ndarray:
    """Create Intrinsic Matrix"""
    # perspective projection (focal length)
    f = w_pers / (2.0 * np.tan(np.radians(fov_x) / 2.0))
    # transform between camera frame and pixel coordinates
    K = np.array(
        [
            [f, skew, w_pers / 2],
            [0.0, f, h_pers / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    return K


def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    z_down: bool = False,
) -> np.ndarray:
    """Create Rotation Matrix"""
    # calculate rotation about the x-axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    # calculate rotation about the y-axis
    if z_down:
        pitch = -pitch
    R_y = np.array(
        [
            [np.cos(pitch), 0.0, -np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    # calculate rotation about the z-axis
    if z_down:
        yaw = -yaw
    R_z = np.array(
        [
            [np.cos(yaw), np.sin(yaw), 0.0],
            [-np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return R_z @ R_y @ R_x


def _create_rotation_matrix(
    x: float,
    y: float,
    z: float,
    z_down: bool = False,
) -> np.ndarray:
    """Create Rotation Matrix

    NOTE: DEPRECATED
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
    if z_down:
        y = -y
    R_y = np.array(
        [
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)],
        ]
    )
    # calculate rotation about the z-axis
    if z_down:
        z = -z
    R_z = np.array(
        [
            [np.cos(z), -np.sin(z), 0.0],
            [np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return R_z @ R_y @ R_x
