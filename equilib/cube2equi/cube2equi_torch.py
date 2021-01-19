#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple, Union

import torch

from equilib.grid_sample import torch_func
from equilib.common.torch_utils import get_device


def cube_list2h(cube_list: List[torch.Tensor]) -> torch.Tensor:
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return torch.cat(cube_list, dim=-1)


def cube_dict2h(
    cube_dict: Dict[str, torch.Tensor],
    face_k: Union[Tuple[str], List[str]] = ("F", "R", "B", "L", "U", "D"),
) -> torch.Tensor:
    assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_dice2h(cube_dice: torch.Tensor) -> torch.Tensor:
    r"""dice to horizion

    params:
    - cube_dice: (C, H, W)
    """
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    w = cube_dice.shape[-2] // 3
    assert cube_dice.shape[-2] == w * 3 and cube_dice.shape[-1] == w * 4
    cube_h = torch.zeros((cube_dice.shape[-3], w, w * 6), dtype=cube_dice.dtype)
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[:, sy * w : (sy + 1) * w, sx * w : (sx + 1) * w]
        cube_h[:, :, i * w : (i + 1) * w] = face
    return cube_h


def _to_horizon(
    cubemap: torch.Tensor,
    cube_format: str,
) -> torch.Tensor:
    if cube_format == "horizon":
        pass
    elif cube_format == "list":
        cubemap = cube_list2h(cubemap)
    elif cube_format == "dict":
        cubemap = cube_dict2h(cubemap)
    elif cube_format == "dice":
        cubemap = cube_dice2h(cubemap)
    else:
        raise NotImplementedError("unknown cube_format")

    assert len(cubemap.shape) == 3
    assert cubemap.shape[-2] * 6 == cubemap.shape[-1]

    return cubemap


def _equirect_facetype(h: int, w: int) -> torch.Tensor:
    r"""0F 1R 2B 3L 4U 5D"""
    tp = torch.roll(
        torch.arange(4)  # 1
        .repeat_interleave(w // 4)  # 2 same as np.repeat
        .unsqueeze(0)
        .transpose(0, 1)  # 3
        .repeat(1, h)  # 4
        .view(-1, h)  # 5
        .transpose(0, 1),  # 6
        shifts=3 * w // 8,
        dims=1,
    )

    # Prepare ceil mask
    mask = torch.zeros((h, w // 4), dtype=torch.bool)
    idx = torch.linspace(-math.pi, math.pi, w // 4) / 4
    idx = h // 2 - torch.round(torch.atan(torch.cos(idx)) * h / math.pi).type(
        torch.int
    )
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = torch.roll(torch.cat([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[torch.flip(mask, dims=(0,))] = 5

    return tp.type(torch.int32)


def create_equi_grid(h_out: int, w_out: int) -> Tuple[torch.Tensor]:
    _dtype = torch.float32
    theta = torch.linspace(-math.pi, math.pi, steps=w_out, dtype=_dtype)
    phi = torch.linspace(math.pi, -math.pi, steps=h_out, dtype=_dtype) / 2
    phi, theta = torch.meshgrid([phi, theta])
    return theta, phi


def run(
    cubemap: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
    cube_format: str,
    w_out: int,
    h_out: int,
    sampling_method: str,
    mode: str,
) -> torch.Tensor:
    r"""Run cube to equirectangular image transformation

    params:
    - cubemap: np.ndarray
    - cube_format: ('dice', 'horizon', 'list', 'dict')
    - sampling_method: str
    - mode: str
    """

    # Convert all cubemap format to `horizon` and batched
    if cube_format in ["dice", "horizon"]:
        if isinstance(cubemap, torch.Tensor):
            assert (
                len(cubemap.shape) >= 3
            ), "input shape {} is not valid".format(cubemap.shape)
            if len(cubemap.shape) == 4:
                cubemap = [_to_horizon(c, cube_format) for c in cubemap]
            else:
                cubemap = [_to_horizon(cubemap, cube_format)]
        elif isinstance(cubemap, list):
            cubemap = [_to_horizon(c, cube_format) for c in cubemap]
        else:
            raise ValueError
    elif cube_format == "dict":
        if isinstance(cubemap, dict):
            cubemap = [_to_horizon(cubemap, cube_format)]
        elif isinstance(cubemap, list):
            cubemap = [_to_horizon(c, cube_format) for c in cubemap]
        else:
            raise ValueError
    elif cube_format == "list":
        if isinstance(cubemap[0], torch.Tensor):
            cubemap = [_to_horizon(cubemap, cube_format)]
        elif isinstance(cubemap[0], list):
            cubemap = [_to_horizon(c, cube_format) for c in cubemap]
        else:
            raise ValueError
    else:
        raise ValueError

    batch = len(cubemap)
    cubemap = torch.stack(cubemap, 0)

    # get device
    device = get_device(cubemap)
    w_face = cubemap.shape[-2]

    theta, phi = create_equi_grid(h_out, w_out)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = _equirect_facetype(h_out, w_out)
    coor_x = torch.zeros((h_out, w_out))
    coor_y = torch.zeros((h_out, w_out))

    # FIXME: there's a bug where left section (3L) has artifacts
    # on top and bottom
    # It might have to do with 4U or 5D
    for i in range(6):
        mask = tp == i

        if i < 4:
            coor_x[mask] = 0.5 * torch.tan(theta[mask] - math.pi * i / 2)
            coor_y[mask] = (
                -0.5
                * torch.tan(phi[mask])
                / torch.cos(theta[mask] - math.pi * i / 2)
            )
        elif i == 4:
            c = 0.5 * torch.tan(math.pi / 2 - phi[mask])
            coor_x[mask] = c * torch.sin(theta[mask])
            coor_y[mask] = c * torch.cos(theta[mask])
        elif i == 5:
            c = 0.5 * torch.tan(math.pi / 2 - torch.abs(phi[mask]))
            coor_x[mask] = c * torch.sin(theta[mask])
            coor_y[mask] = -c * torch.cos(theta[mask])

    # Final renormalize
    coor_x = torch.clamp(
        torch.clamp(coor_x + 0.5, 0, 1) * w_face, 0, w_face - 1
    )
    coor_y = torch.clamp(
        torch.clamp(coor_y + 0.5, 0, 1) * w_face, 0, w_face - 1
    )

    # change x axis of the x coordinate map
    for i in range(6):
        mask = tp == i
        coor_x[mask] = coor_x[mask] + w_face * i

    # repeat batch
    coor_x = coor_x.repeat(batch, 1, 1)
    coor_y = coor_y.repeat(batch, 1, 1)

    grid = torch.stack((coor_y, coor_x), axis=-3).to(device)
    grid_sample = getattr(torch_func, sampling_method)
    equis = grid_sample(cubemap, grid, mode=mode)

    if batch == 1:
        equis = equis.squeeze(0)

    return equis
