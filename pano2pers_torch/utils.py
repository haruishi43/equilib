#!/usr/bin/env python3

from typing import List, Tuple

import torch

pi = torch.Tensor([3.14159265358979323846])


def deg2rad(tensor):
    r"""Function that converts angles from degrees to radians.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.


def create_coord(
    height: int,
    width: int,
) -> torch.tensor:
    r"""Create mesh coordinate grid
    """
    _xs = torch.linspace(0, width-1, width)
    _ys = torch.linspace(0, height-1, height)
    xs, ys = torch.meshgrid(_xs, _ys)
    zs = torch.ones_like(xs)
    coord = torch.stack((xs, ys, zs), dim=2)
    return coord


def create_K(
    height: int, width: int,
    fov_x: float,
) -> torch.tensor:
    f = width / (2 * torch.tan(deg2rad(fov_x) / 2))
    K = torch.tensor([
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]])
    return K


def create_rot_mat(rot: List[float]) -> torch.tensor:
    r"""param: rot: [yaw, pitch, roll]
    """
    rot_yaw, rot_pitch, rot_roll = rot

    R_yaw = torch.tensor([
        [torch.cos(rot_yaw), 0, -torch.sin(rot_yaw)],
        [0, 1, 0],
        [torch.sin(rot_yaw), 0, torch.cos(rot_yaw)]])
    R_pitch = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(rot_pitch), -torch.sin(rot_pitch)],
        [0, torch.sin(rot_pitch), torch.cos(rot_pitch)]])
    R_roll = torch.tensor([
        [torch.cos(rot_roll), -torch.sin(rot_roll), 0],
        [torch.sin(rot_roll), torch.cos(rot_roll), 0],
        [0, 0, 1]])
    R = R_roll @ R_pitch @ R_yaw
    return R


def pixel_wise_rot(rot_coord: np.array) -> Tuple[np.array]:
    a = torch.atan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
    b = torch.asin(rot_coord[:, :, 1] / torch.norm(rot_coord, axis=2))
    return a, b
