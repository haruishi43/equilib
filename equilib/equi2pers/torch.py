#!/usr/bin/env python3

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    create_global2camera_rotation_matrix,
    create_grid,
    create_intrinsic_matrix,
    create_rotation_matrices,
    get_device,
    pi,
)


@lru_cache(maxsize=128)
def create_cam2global_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:

    K = create_intrinsic_matrix(
        height=height,
        width=width,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )
    g2c_rot = create_global2camera_rotation_matrix(dtype=dtype, device=device)

    return g2c_rot @ K.inverse()


def prep_matrices(
    height: int,
    width: int,
    batch: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:

    m = create_grid(
        height=height, width=width, batch=batch, dtype=dtype, device=device
    )
    m = m.unsqueeze(-1)
    G = create_cam2global_matrix(
        height=height,
        width=width,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )

    return m, G


def matmul(m: torch.Tensor, G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:

    M = torch.matmul(torch.matmul(R, G)[:, None, None, ...], m)
    M = M.squeeze(-1)

    return M


def convert_grid(
    M: torch.Tensor, h_equi: int, w_equi: int, method: str = "robust"
) -> torch.Tensor:

    # convert to rotation
    phi = torch.asin(M[..., 2] / torch.norm(M, dim=-1))
    theta = torch.atan2(M[..., 1], M[..., 0])

    if method == "robust":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui = torch.where(ui < 0, ui + w_equi, ui)
        ui = torch.where(ui >= w_equi, ui - w_equi, ui)
        uj = torch.where(uj < 0, uj + h_equi, uj)
        uj = torch.where(uj >= h_equi, uj - h_equi, uj)
    else:
        raise ValueError(f"ERR: {method} is not supported")

    # stack the pixel maps into a grid
    grid = torch.stack((uj, ui), dim=-3)

    return grid


def run(
    equi: torch.Tensor,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
    mode: str,
    backend: str = "native",
) -> torch.Tensor:
    """Run Equi2Pers

    params:
    - equi (torch.Tensor): 4 dims (b, c, h, w)
    - rots (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - height, width (int): height and width of perspective view
    - fov_x (float): fov of horizontal axis in degrees
    - skew (float): skew of the camera
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - backend (str): backend of torch `grid_sample` (default: `native`)

    returns:
    - out (torch.Tensor)

    NOTE: `backend` can be either `native` or `pure`

    """

    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        rots
    ), f"ERR: length of equi and rot differs: {len(equi)} vs {len(rots)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if equi.device.type == "cuda":
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and equi_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        equi = equi.type(torch.float32)

    bs, c, h_equi, w_equi = equi.shape
    img_device = get_device(equi)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initialize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, height, width), dtype=dtype, device=img_device
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if equi.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # create grid and transform matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=tmp_dtype,
        device=tmp_device,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device
    )

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    # if backend == "native":
    #     grid = grid.to(img_device)
    # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
    # FIXME: better way of forcing `grid` to be the same dtype?
    if equi.dtype != grid.dtype:
        grid = grid.type(equi.dtype)
    if equi.device != grid.device:
        grid = grid.to(equi.device)

    # grid sample
    out = torch_grid_sample(
        img=equi,
        grid=grid,
        out=out,  # FIXME: is this necessary?
        mode=mode,
        backend=backend,
    )

    # NOTE: we assume that `out` keeps it's dtype

    out = (
        out.type(equi_dtype)
        if equi_dtype == torch.uint8
        else torch.clip(out, 0.0, 1.0)
    )

    return out


def get_bounding_fov(
    equi: torch.Tensor,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
) -> torch.Tensor:
    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        rots
    ), f"ERR: length of equi and rot differs: {len(equi)} vs {len(rots)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if equi.device.type == "cuda":
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )

    bs, c, h_equi, w_equi = equi.shape

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if equi.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # create grid and transform matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=tmp_dtype,
        device=tmp_device,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device
    )

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    bboxs = []

    # top row
    for out_x in range(width):
        bboxs.append(grid[:, :, 0, out_x])

    # right column
    for out_y in range(height):
        if out_y > 0:  # exclude first
            bboxs.append(grid[:, :, out_y, width - 1])

    # bottom row
    for out_x in range(width - 2, 0, -1):
        bboxs.append(grid[:, :, height - 1, out_x])

    # left column
    for out_y in range(height - 1, 0, -1):
        bboxs.append(grid[:, :, out_y, 0])

    assert len(bboxs) == width * 2 + (height - 2) * 2

    bboxs = torch.stack(bboxs, dim=1)

    bboxs = bboxs.numpy()
    bboxs = np.rint(bboxs).astype(np.int64)

    return bboxs
