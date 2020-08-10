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
        w_face = cubemap.shape[-2]

        rot_coord = self.create_equi_grid(self.h_out, self.w_out)
        theta, phi = np.split(rot_coord, 2, axis=-1)
        theta = theta[..., 0]  # shape(h, w)
        phi = phi[..., 0]  # shape(h, w)

        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        tp = self.equirect_facetype(self.h_out, self.w_out)
        coor_x = np.zeros((self.h_out, self.w_out))
        coor_y = np.zeros((self.h_out, self.w_out))

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

        coor_x = np.where(coor_x >= w_face, coor_x - w_face, coor_x)
        coor_y = np.where(coor_y >= w_face, coor_y - w_face, coor_y)

        # FIXME: there are stiching marks in the equirectangular image
        for i in range(6):
            mask = (tp == i)
            coor_x[mask] = coor_x[mask] + w_face * i

            # exceptions
            if i == 3:
                coor_x[mask] = np.where(
                    coor_x[mask] >= w_face*(i+1) - 1,
                    0,
                    coor_x[mask]
                )

            if i < 4:
                coor_y[mask] = np.where(
                    coor_y[mask] == 0,
                    w_face-1,
                    coor_y[mask]
                )
                coor_y[mask] = np.where(
                    coor_y[mask] == w_face - 1,
                    0,
                    coor_y[mask]
                )

        # cube_faces = np.stack(np.split(cubemap, 6, -1), 0)

        # Switcing top/bottom of U/D image
        # tmp = cube_faces[5][:, :1, :].copy()
        # cube_faces[5][:, :1, :] = cube_faces[4][:, :1, :]
        # cube_faces[4][:, :1, :] = tmp
        # cubemap[:, :, w_face*4:w_face*5] = cube_faces[4]
        # cubemap[:, :, w_face*5:w_face*6] = cube_faces[5]

        # Switching top/bottom of F image makes it worse
        # tmp = cube_faces[0][:, :1, :].copy()
        # cube_faces[0][:, :1, :] = cube_faces[0][:, -1:, :]
        # cube_faces[0][:, -1:, :] = tmp
        # cubemap[:, :, 0:w_face] = cube_faces[0]

        grid = np.stack((coor_y, coor_x), axis=0)
        grid_sample = getattr(
            numpy_func,
            sampling_method
        )
        equi = grid_sample(cubemap, grid, mode=mode)

        return equi
