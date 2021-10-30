#!/usr/bin/env python3

from typing import Dict, List

import numpy as np

import torch


def sin(digit):
    return torch.sin(torch.tensor([digit])).item()

def cos(digit):
    return torch.cos(torch.tensor([digit])).item()


def create_global2camera_rotation_matrix(
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Rotation from global (world) to camera coordinates"""
    R_XY = torch.tensor(
        [  # X <-> Y
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    R_YZ = torch.tensor(
        [  # Y <-> Z
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )
    R = R_XY @ R_YZ
    return R.to(device)


def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create Rotation Matrix

    params:
    - roll, pitch, yaw (float): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (torch.dtype): data types

    returns:
    - R (torch.Tensor): 3x3 rotation matrix
    """

    # calculate rotation about the x-axis
    R_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(roll), -sin(roll)],
            [0.0, sin(roll), cos(roll)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the y-axis
    if not z_down:
        pitch = -pitch
    R_y = torch.tensor(
        [
            [cos(pitch), 0.0, sin(pitch)],
            [0.0, 1.0, 0.0],
            [-sin(pitch), 0.0, cos(pitch)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the z-axis
    if not z_down:
        yaw = -yaw
    R_z = torch.tensor(
        [
            [cos(yaw), -sin(yaw), 0.0],
            [sin(yaw), cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    R = R_z @ R_y @ R_x
    return R.to(device)


def create_rotation_matrix_at_once(
    roll: float,
    pitch: float,
    yaw: float,
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create rotation matrix at once"

    params:
    - roll, pitch, yaw (float): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (torch.dtype): data types
    - device (torch.device): torch.device("cpu")

    returns:
    - R (torch.Tensor): 3x3 rotation matrix

    NOTE: same results as `create_rotation_matrix` but a little bit faster
    """

    if not z_down:
        pitch = -pitch
        yaw = -yaw

    return torch.tensor(
        [
            [
                cos(yaw) * cos(pitch),
                cos(yaw) * sin(pitch) * sin(roll)
                - sin(yaw) * cos(roll),
                cos(yaw) * sin(pitch) * cos(roll)
                + sin(yaw) * sin(roll),
            ],
            [
                sin(yaw) * cos(pitch),
                sin(yaw) * sin(yaw) * sin(pitch) * sin(roll)
                + cos(yaw) * cos(roll),
                sin(yaw) * sin(pitch) * cos(roll)
                - cos(yaw) * sin(roll),
            ],
            [
                -sin(pitch),
                cos(pitch) * sin(roll),
                cos(pitch) * cos(roll),
            ],
        ],
        dtype=dtype,
        device=device,
    )


def create_rotation_matrices(
    rots: List[Dict[str, float]],
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create rotation matrices from batch of rotations

    This methods creates a bx3x3 np.ndarray where `b` referes to the number
    of rotations (rots) given in the input
    """

    R = torch.empty((len(rots), 3, 3), dtype=dtype, device=device)
    for i, rot in enumerate(rots):
        # FIXME: maybe default to `create_rotation_matrix_at_once`?
        # NOTE: at_once is faster with cpu, while slower on GPU
        R[i, ...] = create_rotation_matrix(
            **rot,
            z_down=z_down,
            dtype=dtype,
            device=device,
        )

    return R


def create_rotation_matrix_dep(
    x: float,
    y: float,
    z: float,
    z_down: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create Rotation Matrix

    NOTE: DEPRECATED
    """
    # calculate rotation about the x-axis
    R_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(x), -sin(x)],
            [0.0, sin(x), cos(x)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the y-axis
    if not z_down:
        y = -y
    R_y = torch.tensor(
        [
            [cos(y), 0.0, -sin(y)],
            [0.0, 1.0, 0.0],
            [sin(y), 0.0, cos(y)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the z-axis
    if not z_down:
        z = -z
    R_z = torch.tensor(
        [
            [cos(z), sin(z), 0.0],
            [-sin(z), cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    R = R_z @ R_y @ R_x
    return R.to(device)
