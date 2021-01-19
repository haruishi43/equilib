#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple, Union

import numpy as np

import torch

from equilib.grid_sample import torch_func
from equilib.common.torch_utils import (
    create_rotation_matrix,
    deg2rad,
    get_device,
    sizeof,
)


def intrinsic_matrix(
    w_pers: int,
    h_pers: int,
    fov_x: float,
    skew: float,
) -> torch.Tensor:
    r"""Create Intrinsic Matrix

    return:
    - K (torch.Tensor): 3x3 matrix

    NOTE:
    - ref: http://ksimek.github.io/2013/08/13/intrinsic/
    """
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


def perspective_coordinate(
    w_pers: int,
    h_pers: int,
) -> torch.Tensor:
    r"""Create mesh coordinate grid with perspective height and width

    return:
    - coordinate (torch.Tensor)
    """
    _xs = torch.linspace(0, w_pers - 1, w_pers)
    _ys = torch.linspace(0, h_pers - 1, h_pers)
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs])
    zs = torch.ones_like(xs)
    coord = torch.stack((xs, ys, zs), dim=2)
    return coord


def rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    r"""Create Rotation Matrix

    params:
    - roll: x-axis rotation float
    - pitch: y-axis rotation float
    - yaw: z-axis rotation float

    return:
    - rotation matrix (torch.Tensor)

    Global coordinates -> x-axis points forward, z-axis points upward
    """
    R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
    return R


def global2camera_rotation_matrix() -> torch.Tensor:
    r"""Default rotation that changes global to camera coordinates"""
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


def _get_img_size(img: torch.Tensor) -> Tuple[int]:
    r"""Return height and width"""
    # batch, channel, height, width
    return img.shape[-2:]


def run(
    equi: torch.Tensor,
    rot: Union[Dict[str, float], List[Dict[str, float]]],
    w_pers: int,
    h_pers: int,
    fov_x: float,
    skew: float,
    sampling_method: str,
    mode: str,
    debug: bool = False,
) -> torch.Tensor:
    r"""Run Equi2Pers

    params:
    - equi: equirectangular image torch.Tensor[(B), C, H, W]
    - rot: Dict[str, float] or List[Dict[str, float]]
    - sampling_method (str)
    - mode (str)

    returns:
    - pers: perspective image torch.Tensor[C, H, W]

    NOTE: input can be batched [B, C, H, W] or single [C, H, W]
    NOTE: when using batches, the output types match
    """
    assert (
        type(equi) == torch.Tensor
    ), "ERR: input equi expected to be `torch.Tensor` " "but got {}".format(
        type(equi)
    )
    _original_shape_len = len(equi.shape)
    assert _original_shape_len >= 3, "ERR: got {} for input equi".format(
        _original_shape_len
    )
    if _original_shape_len == 3:
        equi = equi.unsqueeze(dim=0)
        rot = [rot]

    h_equi, w_equi = _get_img_size(equi)
    if debug:
        print("equi: ", sizeof(equi) / 10e6, "mb")

    # get device
    device = get_device(equi)

    # define variables
    # FIXME: methods without using loop
    M = []
    for r in rot:
        # for each rotations calculate M
        m = perspective_coordinate(
            w_pers=w_pers,
            h_pers=h_pers,
        )
        K = intrinsic_matrix(
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
            skew=skew,
        )
        g2c_rot = global2camera_rotation_matrix()
        R = rotation_matrix(**r)
        _M = R @ g2c_rot @ K.inverse() @ m.unsqueeze(3)
        _M = _M.squeeze(3)
        M.append(_M)
    M = torch.stack(M, dim=0).to(device)

    # calculate rotations per perspective coordinates
    norms = torch.norm(M, dim=-1)
    phi = torch.asin(M[:, :, :, 2] / norms)
    theta = torch.atan2(M[:, :, :, 1], M[:, :, :, 0])

    # center the image and convert to pixel locatio
    ui = (theta - math.pi) * w_equi / (2 * math.pi)
    uj = (phi - math.pi / 2) * h_equi / math.pi
    # out-of-bounds calculations
    ui = torch.where(ui < 0, ui + w_equi, ui)
    ui = torch.where(ui >= w_equi, ui - w_equi, ui)
    uj = torch.where(uj < 0, uj + h_equi, uj)
    uj = torch.where(uj >= h_equi, uj - h_equi, uj)
    grid = torch.stack((uj, ui), axis=-3)  # 3rd to last

    # grid sample
    grid_sample = getattr(torch_func, sampling_method, "default")
    samples = grid_sample(equi, grid, mode=mode)

    if _original_shape_len == 3:
        samples = samples.squeeze(axis=0)

    return samples
