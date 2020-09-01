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


def cube_h2list(cube_h):
    assert cube_h.shape[-2] * 6 == cube_h.shape[-1]
    return torch.split(cube_h, split_size_or_sections=cube_h.shape[-2], dim=-1)


def cube_h2dict(cube_h):
    cube_list = cube_h2list(cube_h)
    if len(cube_h.shape) == 3:
        return dict(
            [
                (k, cube_list[i])
                for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
            ]
        )
    else:
        batches = cube_h.shape[0]
        ret = []
        for b in range(batches):
            ret.append(
                dict(
                    [
                        (k, cube_list[i][b])
                        for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
                    ]
                )
            )
        return ret


def cube_h2dice(cube_h):
    w = cube_h.shape[-2]
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]

    if len(cube_h.shape) == 3:
        batches = 1
        cube_dice = torch.zeros(
            (cube_h.shape[-3], w * 3, w * 4), dtype=cube_h.dtype
        )
        cube_list = cube_h2list(cube_h)
        for i, (sx, sy) in enumerate(sxy):
            cube_dice[
                :, sy * w : (sy + 1) * w, sx * w : (sx + 1) * w
            ] = cube_list[i]
    else:
        batches = cube_h.shape[0]
        cube_dice = torch.zeros(
            (batches, cube_h.shape[-3], w * 3, w * 4), dtype=cube_h.dtype
        )
        cube_list = cube_h2list(cube_h)
        for b in range(batches):
            for i, (sx, sy) in enumerate(sxy):
                cube_dice[
                    b, :, sy * w : (sy + 1) * w, sx * w : (sx + 1) * w
                ] = cube_list[i][b]

    return cube_dice
