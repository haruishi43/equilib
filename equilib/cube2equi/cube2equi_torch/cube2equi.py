#!/usr/bin/env python3

from typing import Dict, List, Union

import math
import torch

from equilib.grid_sample import torch_func

from .utils import (
    get_device,
    sizeof
)
from ..base import BaseCube2Equi


def cube_list2h(cube_list: list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return torch.cat(cube_list, dim=-1)


def cube_dict2h(cube_dict: dict, face_k=['F', 'R', 'B', 'L', 'U', 'D']):
    assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_dice2h(cube_dice: torch.Tensor):
    r"""dice to horizion
    params:
    cube_dice: (C, H, W)
    """
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    w = cube_dice.shape[-2] // 3
    assert cube_dice.shape[-2] == w * 3 and cube_dice.shape[-1] == w * 4
    cube_h = torch.zeros(
        (cube_dice.shape[-3], w, w * 6),
        dtype=cube_dice.dtype
    )
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[:, sy*w:(sy+1)*w, sx*w:(sx+1)*w]
        cube_h[:, :, i*w:(i+1)*w] = face
    return cube_h


class Cube2Equi(BaseCube2Equi):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _to_horizon(
        self,
        cubemap: torch.Tensor,
        cube_format: str,
    ) -> torch.Tensor:
        if cube_format == 'horizon':
            pass
        elif cube_format == 'list':
            cubemap = cube_list2h(cubemap)
        elif cube_format == 'dict':
            cubemap = cube_dict2h(cubemap)
        elif cube_format == 'dice':
            cubemap = cube_dice2h(cubemap)
        else:
            raise NotImplementedError('unknown cube_format')

        assert len(cubemap.shape) == 3
        assert cubemap.shape[-2] * 6 == cubemap.shape[-1]

        return cubemap

    def _equirect_facetype(self, h: int, w: int) -> torch.Tensor:
        r"""0F 1R 2B 3L 4U 5D
        """
        tp = torch.roll(
            torch.arange(4)
            .repeat(w // 4)
            .unsqueeze(0)
            .transpose(0, 1)
            .repeat(1, h).view(-1, h)
            .transpose(0, 1),
            shifts=3 * w // 8,
            dims=1
        )

        # Prepare ceil mask
        mask = torch.zeros((h, w // 4), dtype=torch.bool)
        idx = torch.linspace(-math.pi, math.pi, w // 4) / 4
        idx = h // 2 \
            - torch.round(
                torch.atan(
                    torch.cos(idx)
                ) * h / math.pi
            ).type(torch.int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = torch.roll(
            torch.cat(
                [mask] * 4,
                1
            ),
            3 * w // 8,
            1
        )

        tp[mask] = 4
        tp[torch.flip(mask, dims=(0,))] = 5

        return tp.type(torch.int32)

    def create_equi_grid(self, h_out: int, w_out: int) -> torch.Tensor:
        _dtype = torch.float32
        theta = torch.linspace(-math.pi, math.pi, steps=w_out, dtype=_dtype)
        phi = torch.linspace(math.pi, -math.pi, steps=h_out, dtype=_dtype) / 2
        phi, theta = torch.meshgrid([phi, theta])
        return theta, phi

    def run(
        self,
        cubemap: Union[torch.Tensor, dict, list],
        cube_format: str = 'dice',
        sampling_method: str = 'torch',
        mode: str = 'bilinear',
    ) -> torch.Tensor:
        r"""
        """
        # get device
        device = get_device(cubemap)
        w_face = cubemap.shape[-2]

        theta, phi = self.create_equi_grid(self.h_out, self.w_out)

        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        tp = self._equirect_facetype(self.h_out, self.w_out)
        coor_x = torch.zeros((self.h_out, self.w_out))
        coor_y = torch.zeros((self.h_out, self.w_out))

        for i in range(4):
            mask = (tp == i)
            coor_x[mask] = 0.5 * torch.tan(theta[mask] - math.pi * i / 2)
            coor_y[mask] = -0.5 * torch.tan(
                phi[mask]) / torch.cos(theta[mask] - math.pi * i / 2)

        mask = (tp == 4)
        c = 0.5 * torch.tan(math.pi / 2 - phi[mask])
        coor_x[mask] = c * torch.sin(theta[mask])
        coor_y[mask] = c * torch.cos(theta[mask])

        # Final renormalize
        coor_x = (torch.clamp(coor_x, -0.5, 0.5) + 0.5) * w_face
        coor_y = (torch.clamp(coor_y, -0.5, 0.5) + 0.5) * w_face

        coor_x = torch.where(coor_x >= w_face, coor_x - w_face, coor_x)
        coor_y = torch.where(coor_y >= w_face, coor_y - w_face, coor_y)

        # FIXME: there are stiching marks in the equirectangular image
        # zero = torch.tensor(0, dtype=torch.float)
        # edge = torch.tensor(w_face-1, dtype=torch.float)
        # for i in range(6):
        #     mask = (tp == i)
        #     if i == 4:
        #         coor_x[mask] += 1
        #         coor_x[mask] = torch.where(
        #             coor_x[mask] > edge, edge, coor_x[mask])

        #     if i == 5:
        #         coor_x[mask] -= 1
        #         coor_x[mask] = torch.where(
        #             coor_x[mask] < zero, zero, coor_x[mask]
        #         )

        #     coor_x[mask] = coor_x[mask] + w_face * i

        #     # exceptions
        #     if i == 3:
        #         coor_x[mask] = torch.where(
        #             coor_x[mask] >= w_face*(i+1) - 1,
        #             zero,
        #             coor_x[mask]
        #         )

        #     if i < 4:
        #         coor_y[mask] -= 1
        #         coor_y[mask] = torch.where(coor_y[mask] < zero, zero, coor_y[mask])

        #     if i == 4:
        #         coor_y[mask] -= 1
        #         coor_y[mask] = torch.where(coor_y[mask] < zero, zero, coor_y[mask])
        #     if i == 5:
        #         coor_y[mask] += 1
        #         coor_y[mask] = torch.where(
        #             coor_y[mask] > edge, edge, coor_y[mask])

        grid = torch.stack((coor_y, coor_x), axis=0).to(device)
        grid_sample = getattr(
            torch_func,
            sampling_method
        )
        equi = grid_sample(cubemap, grid, mode=mode)

        return equi
