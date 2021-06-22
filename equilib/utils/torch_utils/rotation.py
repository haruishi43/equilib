#!/usr/bin/env python3

import numpy as np

import torch

pi = torch.Tensor([3.14159265358979323846])


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    """Function that converts angles from degrees to radians"""
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def create_global2camera_rotation_matrix() -> torch.Tensor:
    """Rotation from global (world) to camera coordinates"""
    R_XY = torch.tensor(
        [  # X <-> Y
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R_YZ = torch.tensor(
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
) -> torch.Tensor:
    """Create Intrinsic Matrix"""
    fov_x = torch.tensor(fov_x)
    f = w_pers / (2 * torch.tan(deg2rad(fov_x) / 2))
    K = torch.tensor(
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
) -> torch.Tensor:
    """Create Rotation Matrix"""
    # calculate rotation about the x-axis
    R_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=torch.float,
    )
    # calculate rotation about the y-axis
    if z_down:
        pitch = -pitch
    R_y = torch.tensor(
        [
            [np.cos(pitch), 0.0, -np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=torch.float,
    )
    # calculate rotation about the z-axis
    if z_down:
        yaw = -yaw
    R_z = torch.tensor(
        [
            [np.cos(yaw), np.sin(yaw), 0.0],
            [-np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float,
    )

    return R_z @ R_y @ R_x


def _create_rotation_matrix(
    x: float,
    y: float,
    z: float,
    z_down: bool = False,
) -> torch.Tensor:
    """Create Rotation Matrix

    NOTE: DEPRECATED
    """
    # calculate rotation about the x-axis
    R_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x), -np.sin(x)],
            [0.0, np.sin(x), np.cos(x)],
        ],
        dtype=torch.float,
    )
    # calculate rotation about the y-axis
    if z_down:
        y = -y
    R_y = torch.tensor(
        [
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)],
        ],
        dtype=torch.float,
    )
    # calculate rotation about the z-axis
    if z_down:
        z = -z
    R_z = torch.tensor(
        [
            [np.cos(z), -np.sin(z), 0.0],
            [np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float,
    )

    return R_z @ R_y @ R_x
