#!/usr/bin/env python3

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


def create_coord(
    height: int,
    width: int,
) -> torch.Tensor:
    r"""Create mesh coordinate grid"""
    _xs = torch.linspace(0, width - 1, width)
    _ys = torch.linspace(0, height - 1, height)
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs])
    zs = torch.ones_like(xs)
    coord = torch.stack((xs, ys, zs), dim=2)
    return coord


def create_rotation_matrix(
    x: float,
    y: float,
    z: float,
) -> torch.Tensor:
    r"""Create Rotation Matrix

    params:
    - x: x-axis rotation float
    - y: y-axis rotation float
    - z: z-axis rotation float

    return:
    - rotation matrix (torch.Tensor)
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
