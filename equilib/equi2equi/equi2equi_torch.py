#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple, Union

import numpy as np

import torch

from equilib.grid_sample import torch_func
from equilib.common.torch_utils import (
    create_rotation_matrix,
    get_device,
    sizeof,
)


def pixel_wise_rot(M: torch.Tensor) -> Tuple[torch.Tensor]:
    r"""Rotation coordinates to phi/theta of the equirectangular image

    params:
    - M: torch.Tensor

    return:
    - phis: torch.Tensor
    - thetas: torch.Tensor
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


def create_coordinate(h_out: int, w_out: int) -> np.ndarray:
    r"""Create mesh coordinate grid with height and width

    return:
    - coordinate (np.ndarray)
    """
    xs = torch.linspace(0, w_out - 1, w_out)
    theta = xs * 2 * math.pi / w_out - math.pi
    ys = torch.linspace(0, h_out - 1, h_out)
    phi = ys * math.pi / h_out - math.pi / 2
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    phi, theta = torch.meshgrid([phi, theta])
    coord = torch.stack((theta, phi), axis=-1)
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
    - rotation matrix (np.ndarray)

    Global coordinates -> x-axis points forward, z-axis points upward
    """
    R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
    return R


def _get_img_size(img: torch.Tensor) -> Tuple[int]:
    r"""Return height and width"""
    # batch, channel, height, width
    return img.shape[-2:]


def run(
    src: torch.Tensor,
    rot: Union[Dict[str, float], List[Dict[str, float]]],
    w_out: int,
    h_out: int,
    sampling_method: str = "torch",
    mode: str = "bilinear",
    debug: bool = False,
) -> torch.Tensor:
    r"""Run Equi2Pers

    params:
    - src: equirectangular image torch.Tensor[(B), C, H, W]
    - rot: Dict[str, float] or List[Dict[str, float]]
    - sampling_method (str): (default="torch")
    - mode (str): (default="bilinear")

    returns:
    - pers: perspective image torch.Tensor[C, H, W]

    NOTE: input can be batched [B, C, H, W] or single [C, H, W]
    NOTE: when using batches, the output types match
    """
    assert (
        type(src) == torch.Tensor
    ), "ERR: input equi expected to be `torch.Tensor` " "but got {}".format(
        type(src)
    )
    _original_shape_len = len(src.shape)
    assert _original_shape_len >= 3, "ERR: got {} for input equi".format(
        _original_shape_len
    )
    if _original_shape_len == 3:
        src = src.unsqueeze(dim=0)
        rot = [rot]

    h_equi, w_equi = _get_img_size(src)
    if h_out is None and w_out is None:
        h_out = h_equi
        w_out = w_equi

    if debug:
        print("size of src: ", sizeof(src) / 10e6, "mb")

    # get device
    device = get_device(src)

    # define variables
    B = []
    for r in rot:
        a = create_coordinate(h_out, w_out)
        norm_A = 1
        x = norm_A * torch.cos(a[:, :, 1]) * torch.cos(a[:, :, 0])
        y = norm_A * torch.cos(a[:, :, 1]) * torch.sin(a[:, :, 0])
        z = norm_A * torch.sin(a[:, :, 1])
        A = torch.stack((x, y, z), dim=-1)
        R = rotation_matrix(**r)
        _B = R @ A.unsqueeze(3)
        _B = _B.squeeze(3)
        B.append(_B)
    B = torch.stack(B, dim=0).to(device)

    # calculate rotations per perspective coordinates
    norms = torch.norm(B, dim=-1)
    theta = torch.atan2(B[:, :, :, 1], B[:, :, :, 0])
    phi = torch.asin(B[:, :, :, 2] / norms)

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
    grid_sample = getattr(torch_func, sampling_method, "torch")
    samples = grid_sample(src, grid, mode=mode)

    if _original_shape_len == 3:
        samples = samples.squeeze(axis=0)

    return samples
