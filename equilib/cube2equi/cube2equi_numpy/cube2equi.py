#!/usr/bin/env python3

import numpy as np

from equilib.grid_sample import numpy_func

from .utils import (
    equirect_uvgrid,
    equirect_facetype,
    sample_cubefaces,
    cube_list2h,
    cube_dict2h,
    cube_dice2h,
)
from ..base import BaseCube2Equi


class Cube2Equi(BaseCube2Equi):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, cubemap, h, w, mode='bilinear', cube_format='dice'):
        if mode == 'bilinear':
            order = 1
        elif mode == 'nearest':
            order = 0
        else:
            raise NotImplementedError('unknown mode')

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
        assert cubemap.shape[0] * 6 == cubemap.shape[1]
        assert w % 8 == 0
        face_w = cubemap.shape[0]

        uv = equirect_uvgrid(h, w)
        u, v = np.split(uv, 2, axis=-1)
        u = u[..., 0]
        v = v[..., 0]
        cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        tp = equirect_facetype(h, w)
        coor_x = np.zeros((h, w))
        coor_y = np.zeros((h, w))

        for i in range(4):
            mask = (tp == i)
            coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
            coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

        mask = (tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - v[mask])
        coor_x[mask] = c * np.sin(u[mask])
        coor_y[mask] = c * np.cos(u[mask])

        mask = (tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
        coor_x[mask] = c * np.sin(u[mask])
        coor_y[mask] = -c * np.cos(u[mask])

        # Final renormalize
        coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
        coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

        equirec = np.stack([
            sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
            for i in range(cube_faces.shape[3])
        ], axis=-1)

        return equirec
