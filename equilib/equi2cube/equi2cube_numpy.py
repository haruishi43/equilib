#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

import numpy as np

from equilib.grid_sample import numpy_func
from equilib.common.numpy_utils import create_rotation_matrix


def cube_h2list(cube_h: np.ndarray) -> List[np.ndarray]:
    assert cube_h.shape[-2] * 6 == cube_h.shape[-1]
    return np.split(cube_h, 6, axis=-1)


def cube_h2dict(cube_h: np.ndarray) -> Dict[str, np.ndarray]:
    cube_list = cube_h2list(cube_h)
    return {
        k: cube_list[i] for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
    }


def cube_h2dice(cube_h: np.ndarray) -> np.ndarray:
    assert cube_h.shape[-2] * 6 == cube_h.shape[-1]
    w = cube_h.shape[-2]
    cube_dice = np.zeros((cube_h.shape[0], w * 3, w * 4), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        cube_dice[:, sy * w : (sy + 1) * w, sx * w : (sx + 1) * w] = cube_list[
            i
        ]
    return cube_dice


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


def create_xyz(w_face: int) -> np.ndarray:
    r"""xyz coordinates of the faces of the cube"""
    out = np.zeros((w_face, w_face * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=w_face, dtype=np.float32)

    # Front face (x = 0.5)
    out[:, 0 * w_face : 1 * w_face, [1, 2]] = np.stack(
        np.meshgrid(rng, -rng), -1
    )
    out[:, 0 * w_face : 1 * w_face, 0] = 0.5

    # Right face (y = -0.5)
    out[:, 1 * w_face : 2 * w_face, [0, 2]] = np.stack(
        np.meshgrid(-rng, -rng), -1
    )
    out[:, 1 * w_face : 2 * w_face, 1] = 0.5

    # Back face (x = -0.5)
    out[:, 2 * w_face : 3 * w_face, [1, 2]] = np.stack(
        np.meshgrid(-rng, -rng), -1
    )
    out[:, 2 * w_face : 3 * w_face, 0] = -0.5

    # Left face (y = 0.5)
    out[:, 3 * w_face : 4 * w_face, [0, 2]] = np.stack(
        np.meshgrid(rng, -rng), -1
    )
    out[:, 3 * w_face : 4 * w_face, 1] = -0.5

    # Up face (z = 0.5)
    out[:, 4 * w_face : 5 * w_face, [1, 0]] = np.stack(
        np.meshgrid(rng, rng), -1
    )
    out[:, 4 * w_face : 5 * w_face, 2] = 0.5

    # Down face (z = -0.5)
    out[:, 5 * w_face : 6 * w_face, [1, 0]] = np.stack(
        np.meshgrid(rng, -rng), -1
    )
    out[:, 5 * w_face : 6 * w_face, 2] = -0.5

    return out


def xyz2rot(xyz) -> Tuple[np.ndarray]:
    r"""Return rotation (theta, phi) from xyz"""
    norm = np.linalg.norm(xyz, axis=-1)
    phi = np.arcsin(xyz[:, :, 2] / norm)
    theta = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
    return theta, phi


def _run_single(
    equi: np.ndarray,
    rot: Dict[str, float],
    w_face: int,
    cube_format: str,
    sampling_method: str,
    mode: str,
) -> np.ndarray:
    r"""Call a single run

    params:
    - equi: np.ndarray
    - rot: Dict[str, float]
    - w_face: int
    - cube_format: str
    - mode: str
    """

    assert len(equi.shape) == 3, "ERR: {} is not a valid array".format(
        equi.shape
    )
    assert equi.shape[0] == 3, "ERR: got {} for channel size".format(
        equi.shape[0]
    )
    h_equi, w_equi = equi.shape[-2:]

    xyz = create_xyz(w_face)
    xyz = xyz[:, :, :, np.newaxis]
    xyz_ = rotation_matrix(**rot) @ xyz
    xyz_ = xyz_.squeeze(-1)
    theta, phi = xyz2rot(xyz_)

    # center the image and convert to pixel location
    ui = (theta - np.pi) * w_equi / (2 * np.pi)
    uj = (np.pi / 2 - phi) * h_equi / np.pi

    # out-of-bounds calculations
    ui = np.where(ui < 0, ui + w_equi, ui)
    ui = np.where(ui >= w_equi, ui - w_equi, ui)
    uj = np.where(uj < 0, uj + h_equi, uj)
    uj = np.where(uj >= h_equi, uj - h_equi, uj)
    grid = np.stack((uj, ui), axis=0)

    grid_sample = getattr(numpy_func, sampling_method, "faster")
    cubemap = grid_sample(equi, grid, mode=mode)

    if cube_format == "horizon":
        pass
    elif cube_format == "list":
        cubemap = cube_h2list(cubemap)
    elif cube_format == "dict":
        cubemap = cube_h2dict(cubemap)
    elif cube_format == "dice":
        cubemap = cube_h2dice(cubemap)
    else:
        raise NotImplementedError("{} is not supported".format(cube_format))

    return cubemap


def run(
    equi: Union[np.ndarray, List[np.ndarray]],
    rot: Union[Dict[str, float], List[Dict[str, float]]],
    w_face: int,
    cube_format: str,
    sampling_method: str,
    mode: str,
) -> Union[np.ndarray, List[np.ndarray], List[Dict[str, np.ndarray]]]:
    r"""Call Equi2Cube

    params:
    - equi (Union[np.ndarray, List[np.ndarray]])
    - rot (Union[Dict[str, float], List[Dict[str, float]]])
    - w_face (int)
    - cube_format (str): ('list', 'dict', 'dice')
    - sampling_method (str)
    - mode (str)
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
    ), "ERR: length of input and rot differs {} vs {}".format(
        len(equi), len(rot)
    )

    cubemaps = []
    for e, r in zip(equi, rot):
        # iterate through batches
        # TODO: batch implementation
        cubemap = _run_single(
            equi=e,
            rot=r,
            w_face=w_face,
            cube_format=cube_format,
            sampling_method=sampling_method,
            mode=mode,
        )
        cubemaps.append(cubemap)

    if _return_type == np.ndarray:
        if cube_format in ["horizon", "dice"]:
            cubemaps = np.stack(cubemaps, axis=0)

    if _original_shape_len == 3:
        cubemaps = cubemaps[0]

    return cubemaps
