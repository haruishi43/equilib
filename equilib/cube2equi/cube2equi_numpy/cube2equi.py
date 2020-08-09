#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

from equilib.grid_sample import numpy_func

from .utils import (
    create_rotation_matrix,
    cube_list2h,
    cube_dict2h,
    cube_dice2h,
)
from ..base import BaseCube2Equi


class Cube2Equi(BaseCube2Equi):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _convert_cubemap_to_horizon(
        self,
        cubemap: np.ndarray,
        cube_format: str
    ) -> np.ndarray:
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

    def equirect_facetype(self, h: int, w: int):
        r"""0F 1R 2B 3L 4U 5D
        """
        tp = np.roll(
            np.arange(4).repeat(w // 4)[None, :].repeat(h, 0),
            3 * w // 8,
            1
        )

        # Prepare ceil mask
        mask = np.zeros((h, w // 4), np.bool)
        idx = np.linspace(-np.pi, np.pi, w // 4) / 4
        idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(
            np.concatenate(
                [mask] * 4, 1
            ),
            3 * w // 8,
            1
        )

        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        return tp.astype(np.int32)

    def create_equi_grid(self, h_out: int, w_out: int) -> np.ndarray:
        _dtype = np.float32
        theta = np.linspace(-np.pi, np.pi, num=w_out, dtype=_dtype)
        phi = np.linspace(np.pi, -np.pi, num=h_out, dtype=_dtype) / 2
        coord = np.stack(np.meshgrid(theta, phi), axis=-1)
        return coord

    def convert_cubefaces(self, cube_faces):
        # 0F 1R 2B 3L 4U 5D
        cf = cube_faces.copy()

        # FIXME: what's going on here

        print('cf', cf.shape)  # (6, 3, h, w)

        # cf[1] = np.flip(cube_faces[1], 1)
        # cf[2] = np.flip(cube_faces[2], 1)
        # cf[4] = np.flip(cube_faces[4], 0)

        # Pad up down
        pad_ud = np.zeros((cf.shape[-3], 6, 2))
        pad_ud[:, 0, 0] = cf[:, 5, 0]
        pad_ud[:, 0, 1] = cf[:, 4, -1]
        pad_ud[:, 1, 0] = cf[:, 5, -1]
        pad_ud[:, 1, 1] = cf[-1, 4, ::-1]
        pad_ud[:, 2, 0] = cf[::-1, 5, -1]
        pad_ud[:, 2, 1] = cf[::-1, 4, 0]
        pad_ud[:, 3, 0] = cf[0, 5, ::-1]
        pad_ud[:, 3, 1] = cf[0, 4, :]
        pad_ud[:, 4, 0] = cf[0, 0, :]
        pad_ud[:, 4, 1] = cf[2, 0, ::-1]
        pad_ud[:, 5, 0] = cf[2, -1, ::-1]
        pad_ud[:, 5, 1] = cf[0, -1, :]
        cf = np.concatenate([cf, pad_ud], 1)

        # Pad left right
        pad_lr = np.zeros((6, cf.shape[1], 2))
        pad_lr[0, :, 0] = cf[1, :, 0]
        pad_lr[0, :, 1] = cf[3, :, -1]
        pad_lr[1, :, 0] = cf[2, :, 0]
        pad_lr[1, :, 1] = cf[0, :, -1]
        pad_lr[2, :, 0] = cf[3, :, 0]
        pad_lr[2, :, 1] = cf[1, :, -1]
        pad_lr[3, :, 0] = cf[0, :, 0]
        pad_lr[3, :, 1] = cf[2, :, -1]
        pad_lr[4, 1:-1, 0] = cf[1, 0, ::-1]
        pad_lr[4, 1:-1, 1] = cf[3, 0, :]
        pad_lr[5, 1:-1, 0] = cf[1, -2, :]
        pad_lr[5, 1:-1, 1] = cf[3, -2, ::-1]
        cf = np.concatenate([cf, pad_lr], 2)

        return cf

    def rotation_matrix(
        self,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        r"""Create Rotation Matrix

        params:
            roll: x-axis rotation float
            pitch: y-axis rotation float
            yaw: z-axis rotation float

        return:
            rotation matrix: numpy.ndarray

        Global coordinates -> x-axis points forward, z-axis points upward
        """
        R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
        return R

    def run(
        self,
        cubemap: Union[np.ndarray, dict, list],
        rot: Union[List[dict], dict],  # FIXME: don't really need
        cube_format: str = 'dice',
        sampling_method: str = 'faster',
        mode: str = 'bilinear',
    ) -> np.ndarray:
        r"""Run cube to equirectangular image transformation

        params:
            cubemap: np.ndarray
            cube_format: ('dice', 'horizon', 'list', 'dict')
            sampling_method: str
            mode: str
        """

        cubemap = self._convert_cubemap_to_horizon(cubemap, cube_format)
        cube_faces = np.split(cubemap, 6, -1)
        cube_faces = np.stack(cube_faces, 0)  # (6, 3, h, w)
        w_face = cubemap.shape[-2]

        rot_coord = self.create_equi_grid(self.h_out, self.w_out)
        theta, phi = np.split(rot_coord, 2, axis=-1)
        theta = theta[..., 0]  # shape(h, w)
        phi = phi[..., 0]  # shape(h, w)

        print('theta:', theta.shape)
        print('phi:', phi.shape)

        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        tp = self.equirect_facetype(self.h_out, self.w_out)
        coor_x = np.zeros((self.h_out, self.w_out))
        coor_y = np.zeros((self.h_out, self.w_out))

        print('tp:', tp.shape)

        for i in range(4):
            mask = (tp == i)
            coor_x[mask] = 0.5 * np.tan(theta[mask] - np.pi * i / 2)
            coor_y[mask] = -0.5 * np.tan(phi[mask]) / np.cos(theta[mask] - np.pi * i / 2)

        mask = (tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - phi[mask])
        coor_x[mask] = c * np.sin(theta[mask])
        coor_y[mask] = c * np.cos(theta[mask])

        mask = (tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(phi[mask]))
        coor_x[mask] = c * np.sin(theta[mask])
        coor_y[mask] = -c * np.cos(theta[mask])

        # Final renormalize
        coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * w_face
        coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * w_face

        cube_faces = self.convert_cubefaces(
            cube_faces,
        )
        print('cube_faces:', cube_faces.shape)
        grid = np.stack((coor_y, coor_x), axis=0)

        print('grid:', grid.shape)

        grid_sample = getattr(
            numpy_func,
            sampling_method
        )
        equi = grid_sample(cube_faces, grid, mode=mode)

        print('equi', equi.shape)

        return equi
