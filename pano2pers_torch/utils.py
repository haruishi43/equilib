#!/usr/bin/env python3

from typing import List, Tuple

import torch

pi = torch.Tensor([3.14159265358979323846])


def deg2rad(tensor):
    r"""Function that converts angles from degrees to radians.
    """
    if not torch.is_tensor(tensor):
        return tensor * float(pi) / 180.
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.


def create_coord(
    height: int,
    width: int,
    device: torch.device = torch.device('cpu'),
) -> torch.tensor:
    r"""Create mesh coordinate grid
    """
    _xs = torch.linspace(0, width-1, width, device=device)
    _ys = torch.linspace(0, height-1, height, device=device)
    #NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs])
    zs = torch.ones_like(xs, device=device)
    coord = torch.stack((xs, ys, zs), dim=2)
    return coord


def create_K(
    height: int, width: int,
    fov_x: float,
    device: torch.device = torch.device('cpu'),
) -> torch.tensor:
    fov_x = torch.tensor(fov_x, device=device)
    f = width / (2 * torch.tan(deg2rad(fov_x) / 2))
    K = torch.tensor([
        [f, 0., width/2],
        [0., f, height/2],
        [0., 0., 1.]], device=device)
    return K


def create_rot_mat(
    rot: List[float],
    device: torch.device = torch.device('cpu'),
) -> torch.tensor:
    r"""param: rot: [yaw, pitch, roll]
    """
    rot_yaw, rot_pitch, rot_roll = torch.tensor(
        rot, device=device, dtype=torch.float32)

    R_yaw = torch.tensor([
        [torch.cos(rot_yaw), 0., -torch.sin(rot_yaw)],
        [0., 1., 0.],
        [torch.sin(rot_yaw), 0., torch.cos(rot_yaw)]],
        device=device)
    R_pitch = torch.tensor([
        [1., 0., 0.],
        [0., torch.cos(rot_pitch), -torch.sin(rot_pitch)],
        [0., torch.sin(rot_pitch), torch.cos(rot_pitch)]],
        device=device)
    R_roll = torch.tensor([
        [torch.cos(rot_roll), -torch.sin(rot_roll), 0],
        [torch.sin(rot_roll), torch.cos(rot_roll), 0],
        [0, 0, 1]], device=device)
    R = R_roll @ R_pitch @ R_yaw
    return R


def pixel_wise_rot(rot_coord: torch.tensor) -> Tuple[torch.tensor]:
    a = torch.atan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
    b = torch.asin(rot_coord[:, :, 1] / torch.norm(rot_coord, dim=2))
    return a, b
