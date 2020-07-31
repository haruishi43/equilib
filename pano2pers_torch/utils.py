#!/usr/bin/env python3

from typing import Dict, Tuple
import numpy as np
import torch

pi = torch.Tensor([3.14159265358979323846])


def sizeof(tensor: torch.tensor):
    r"""Get the size of a tensor
    """
    assert torch.is_tensor(tensor), "ERR: is not tensor"
    return tensor.element_size() * tensor.nelement()


def deg2rad(tensor: torch.tensor) -> torch.tensor:
    r"""Function that converts angles from degrees to radians.
    """
    if not torch.is_tensor(tensor):
        return tensor * float(pi) / 180.
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.


def create_rot_coord(
    height: int, width: int,
    fov_x: float,
    rot: Dict[str, float],
    device: torch.device = torch.device('cpu')
) -> torch.tensor:
    r"""Create rot coordinates
    """
    coord = create_coord(height, width, device=device)
    K = create_K(height, width, fov_x, device=device)
    R = create_rot_mat(**rot, device=device)
    rot_coord = R.inverse() @ K.inverse() @ coord.unsqueeze(3)
    rot_coord = rot_coord.squeeze(3)
    return rot_coord


def create_coord(
    height: int,
    width: int,
    device: torch.device = torch.device('cpu'),
) -> torch.tensor:
    r"""Create mesh coordinate grid
    """
    _xs = torch.linspace(0, width-1, width)
    _ys = torch.linspace(0, height-1, height)
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs])
    zs = torch.ones_like(xs)
    coord = torch.stack((xs, ys, zs), dim=2)
    return coord.to(device)


def create_K(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.,
    device: torch.device = torch.device('cpu'),
) -> torch.tensor:
    r"""Create Intrinsic Matrix

    params:
        height: int
        width: int
        fov_x: float
        skew: float
        device: torch.device

    return:
        K: 3x3 matrix torch.tensor

    NOTE:
        ref: http://ksimek.github.io/2013/08/13/intrinsic/
    """
    fov_x = torch.tensor(fov_x, device=device)
    f = width / (2 * torch.tan(deg2rad(fov_x) / 2))
    K = torch.tensor([
        [f, skew, width/2],
        [0., f, height/2],
        [0., 0., 1.]], device=device)
    return K


def create_rot_mat(
    roll: float,
    pitch: float,
    yaw: float,
    device: torch.device = torch.device('cpu'),
) -> torch.tensor:
    r"""Create Rotation Matrix

    params:
        roll: x-axis rotation float
        pitch: y-axis rotation float
        yaw: z-axis rotation float

    return:
        rotation matrix: numpy.ndarray

    Camera coordinates -> z-axis points forward, y-axis points upward
    Global coordinates -> x-axis points forward, z-axis poitns upward

    NOTE: https://www.sciencedirect.com/topics/engineering/intrinsic-parameter
    """
    # default rotation that changes global to camera coordinates
    x = np.pi
    y = np.pi
    z = np.pi
    # calculate rotation about the x-axis
    R_x_ = torch.tensor([
        [1., 0., 0.],
        [0., np.cos(x), -np.sin(x)],
        [0., np.sin(x), np.cos(x)]])
    # calculate rotation about the y-axis
    R_y_ = torch.tensor([
        [np.cos(y), 0., np.sin(y)],
        [0., 1., 0.],
        [-np.sin(y), 0., np.cos(y)]])
    # calculate rotation about the z-axis
    R_z_ = torch.tensor([
        [np.cos(z), -np.sin(z), 0.],
        [np.sin(z), np.cos(z), 0.],
        [0., 0., 1.]])

    R = R_z_ @ R_y_ @ R_x_

    # rotation matrix
    # roll: calculate rotation about the x-axis
    R_x = torch.tensor([
        [1., 0., 0.],
        [0., np.cos(roll), -np.sin(roll)],
        [0., np.sin(roll), np.cos(roll)]])
    # pitch: calculate rotation about the y-axis
    R_y = torch.tensor([
        [np.cos(pitch), 0., np.sin(pitch)],
        [0., 1., 0.],
        [-np.sin(pitch), 0., np.cos(pitch)]])
    # yaw: calculate rotation about the z-axis
    R_z = torch.tensor([
        [np.cos(yaw), -np.sin(yaw), 0.],
        [np.sin(yaw), np.cos(yaw), 0.],
        [0., 0., 1.]])

    R = R @ R_z @ R_y @ R_x
    return R.to(device)


def pixel_wise_rot(rot_coord: torch.tensor) -> Tuple[torch.tensor]:
    r"""Rotation coordinates to phi/theta of the panorama image

    params:
        rot_coord: torch.tensor

    return:
        phis: torch.tensor
        thetas: torch.tensor
    """
    if len(rot_coord.shape) == 3:
        rot_coord = rot_coord.unsqueeze(0)

    norms = torch.norm(rot_coord, dim=-1)
    thetas = torch.atan2(rot_coord[:, :, :, 0], rot_coord[:, :, :, 2])
    phis = torch.asin(rot_coord[:, :, :, 1] / norms)
    if thetas.shape[0] == phis.shape[0] == 1:
        thetas = thetas.squeeze(0)
        phis = phis.squeeze(0)
    return phis, thetas
