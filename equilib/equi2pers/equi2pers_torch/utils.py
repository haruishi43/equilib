#!/usr/bin/env python3

from typing import Dict, Tuple

import numpy as np

import torch

pi = torch.Tensor([3.14159265358979323846])


def sizeof(tensor: torch.Tensor) -> float:
    r"""Get the size of a tensor"""
    assert torch.is_tensor(tensor), "ERR: is not tensor"
    return tensor.element_size() * tensor.nelement()


def get_device(a: torch.Tensor) -> torch.device:
    r"""Get device of a Tensor"""
    return torch.device(a.get_device() if a.get_device() >= 0 else "cpu")


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that converts angles from degrees to radians."""
    if not torch.is_tensor(tensor):
        return tensor * float(pi) / 180.0
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def create_M(
    height: int,
    width: int,
    fov_x: float,
    rot: Dict[str, float],
) -> torch.tensor:
    r"""Create M"""
    m = create_coord(height, width)
    K = create_K(height, width, fov_x)
    R = create_rot_mat(**rot)
    M = R.inverse() @ K.inverse() @ m.unsqueeze(3)
    M = M.squeeze(3)
    return M


def create_coord(
    height: int,
    width: int,
) -> torch.tensor:
    r"""Create mesh coordinate grid"""
    _xs = torch.linspace(0, width - 1, width)
    _ys = torch.linspace(0, height - 1, height)
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs])
    zs = torch.ones_like(xs)
    coord = torch.stack((xs, ys, zs), dim=2)
    return coord


def create_K(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
) -> torch.Tensor:
    r"""Create Intrinsic Matrix

    params:
        height: int
        width: int
        fov_x: float
        skew: float

    return:
        K: 3x3 matrix torch.Tensor

    NOTE:
        ref: http://ksimek.github.io/2013/08/13/intrinsic/
    """
    fov_x = torch.tensor(fov_x)
    f = width / (2 * torch.tan(deg2rad(fov_x) / 2))
    K = torch.tensor(
        [[f, skew, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]]
    )
    return K


def create_rotation_matrix(
    x: float,
    y: float,
    z: float,
) -> torch.Tensor:
    r"""Create Rotation Matrix

    params:
        x: x-axis rotation float
        y: y-axis rotation float
        z: z-axis rotation float

    return:
        rotation matrix: torch.Tensor
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
    R_y = torch.tensor(
        [
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)],
        ],
        dtype=torch.float,
    )
    # calculate rotation about the z-axis
    R_z = torch.tensor(
        [
            [np.cos(z), -np.sin(z), 0.0],
            [np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float,
    )

    return R_z @ R_y @ R_x


def create_rot_mat(
    roll: float,
    pitch: float,
    yaw: float,
) -> torch.Tensor:
    r"""Create Rotation Matrix

    params:
        roll: x-axis rotation float
        pitch: y-axis rotation float
        yaw: z-axis rotation float

    return:
        rotation matrix: torch.Tensor

    Camera coordinates -> z-axis points forward, y-axis points upward
    Global coordinates -> x-axis points forward, z-axis poitns upward

    NOTE: https://www.sciencedirect.com/topics/engineering/intrinsic-parameter
    """
    # default rotation that changes global to camera coordinates
    x = np.pi
    y = np.pi
    z = np.pi
    R = create_rotation_matrix(x=x, y=y, z=z)

    # rotation matrix
    R = R @ create_rotation_matrix(x=roll, y=pitch, z=yaw)
    return R


def pixel_wise_rot(M: torch.Tensor) -> Tuple[torch.Tensor]:
    r"""Rotation coordinates to phi/theta of the equirectangular image

    params:
        M: torch.Tensor

    return:
        phis: torch.Tensor
        thetas: torch.Tensor
    """
    if len(M.shape) == 3:
        M = M.unsqueeze(0)

    norms = torch.norm(M, dim=-1)
    thetas = torch.atan2(M[:, :, :, 0], M[:, :, :, 2])
    phis = torch.asin(M[:, :, :, 1] / norms)

    if thetas.shape[0] == phis.shape[0] == 1:
        thetas = thetas.squeeze(0)
        phis = phis.squeeze(0)
    return phis, thetas
