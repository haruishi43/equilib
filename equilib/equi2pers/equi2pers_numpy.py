#!/usr/bin/env python3

from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np

from equilib.grid_sample import numpy_func
from equilib.utils import (
    create_global2camera_rotation_matrix,
    create_intrinsic_matrix,
    create_rotation_matrix,
)

_create_global2camera_rotation_matrix = partial(
    create_global2camera_rotation_matrix, is_torch=False
)
_create_intrinsic_matrix = partial(create_intrinsic_matrix, is_torch=False)
_create_rotation_matrix = partial(create_rotation_matrix, is_torch=False)


def perspective_coordinate(
    w_pers: int,
    h_pers: int,
) -> np.ndarray:
    """Create mesh coordinate grid with perspective height and width

    return:
    - coordinate (np.ndarray)
    """
    _xs = np.linspace(0, w_pers - 1, w_pers)
    _ys = np.linspace(0, h_pers - 1, h_pers)
    xs, ys = np.meshgrid(_xs, _ys)
    zs = np.ones_like(xs)
    coord = np.stack((xs, ys, zs), axis=2)
    return coord


def _get_img_size(img: np.ndarray) -> Tuple[int]:
    """Return height and width"""
    return img.shape[-2:]


def _run_single(
    equi: np.ndarray,
    rot: Dict[str, float],
    w_pers: int,
    h_pers: int,
    fov_x: float,
    skew: float,
    sampling_method: str,
    mode: str,
    z_down: bool,
) -> np.ndarray:

    # NOTE: Precomputable variables
    m = perspective_coordinate(w_pers=w_pers, h_pers=h_pers)
    K = _create_intrinsic_matrix(
        w_pers=w_pers,
        h_pers=h_pers,
        fov_x=fov_x,
        skew=skew,
    )
    g2c_rot = _create_global2camera_rotation_matrix()

    # Compute variables
    R = _create_rotation_matrix(**rot, z_down=z_down)
    h_equi, w_equi = _get_img_size(equi)

    # conversion
    K_inv = np.linalg.inv(K)
    m = m[:, :, :, np.newaxis]
    M = R @ g2c_rot @ K_inv @ m
    M = M.squeeze(3)

    # calculate rotations per perspective coordinates
    # phi = np.arcsin(M[:, :, 1] / np.linalg.norm(M, axis=-1))
    # theta = np.arctan2(M[:, :, 0], M[:, :, 2])
    phi = np.arcsin(M[:, :, 2] / np.linalg.norm(M, axis=-1))
    theta = np.arctan2(M[:, :, 1], M[:, :, 0])

    # center the image and convert to pixel location
    ui = (theta - np.pi) * w_equi / (2 * np.pi)
    uj = (phi - np.pi / 2) * h_equi / np.pi
    # out-of-bounds calculations
    ui = np.where(ui < 0, ui + w_equi, ui)
    ui = np.where(ui >= w_equi, ui - w_equi, ui)
    uj = np.where(uj < 0, uj + h_equi, uj)
    uj = np.where(uj >= h_equi, uj - h_equi, uj)
    grid = np.stack((uj, ui), axis=0)

    # grid sample
    grid_sample = getattr(numpy_func, sampling_method, "default")
    sampled = grid_sample(equi, grid, mode=mode)
    return sampled


def run(
    equi: Union[np.ndarray, List[np.ndarray]],
    rot: Union[Dict[str, float], List[Dict[str, float]]],
    w_pers: int,
    h_pers: int,
    fov_x: float,
    skew: float,
    sampling_method: str,
    mode: str,
    z_down: bool,
) -> np.ndarray:
    """Run Equi2Pers

    params:
    - equi: equirectangular image np.ndarray[C, H, W]
    - rot (dict, list): Dict[str, float]
    - sampling_method (str)
    - mode (str)
    - z_down (bool)

    returns:
    - pers: perspective image np.ndarray[C, H, W]

    NOTE: input can be batched [B, C, H, W] or List[np.ndarray]
    NOTE: when using batches, the output types match
    """
    _return_type = type(equi)
    _original_shape_len = len(equi.shape)
    if _return_type == np.ndarray:
        assert _original_shape_len >= 3, "ERR: got {} for input equi".format(
            _original_shape_len
        )
        if _original_shape_len == 3:
            equi = equi[np.newaxis, :, :, :]
            rot = [rot]

    assert len(equi) == len(
        rot
    ), "ERR: length of equi and rot differs {} vs {}".format(
        len(equi), len(rot)
    )

    samples = []
    for p, r in zip(equi, rot):
        # iterate through batches
        # TODO: batch implementation
        sample = _run_single(
            equi=p,
            rot=r,
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
            skew=skew,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )
        samples.append(sample)

    if _return_type == np.ndarray:
        samples = np.stack(samples, axis=0)
        if _original_shape_len == 3:
            samples = np.squeeze(samples, axis=0)

    return samples
