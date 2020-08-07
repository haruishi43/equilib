#!/usr/bin/env python3

import numpy as np

from equilib.grid_sample import numpy_func

from .utils import (
    xyzcube,
    xyz2uv,
    uv2coor,
    sample_equirec,
    cube_h2list,
    cube_h2dict,
    cube_h2dice,
)
from ..base import BaseEqui2Cube


class Equi2Cube(BaseEqui2Cube):

    def __init__(self, *kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        equi: np.ndarray,
        cube_format: str,
        mode: str = "bilinear",
    ) -> np.ndarray:
        r"""Call Equi2Cube
        """



        assert len(e_img.shape) == 3
        h, w = e_img.shape[-2:]
        if mode == 'bilinear':
            order = 1
        elif mode == 'nearest':
            order = 0
        else:
            raise NotImplementedError('unknown mode')

        xyz = xyzcube(self.w_face)
        uv = xyz2uv(xyz)
        coor_xy = uv2coor(uv, h, w)

        cubemap = np.stack([
            sample_equirec(e_img[..., i], coor_xy, order=order)
            for i in range(e_img.shape[2])
        ], axis=-1)

        if cube_format == 'horizon':
            pass
        elif cube_format == 'list':
            cubemap = cube_h2list(cubemap)
        elif cube_format == 'dict':
            cubemap = cube_h2dict(cubemap)
        elif cube_format == 'dice':
            cubemap = cube_h2dice(cubemap)
        else:
            raise NotImplementedError()

        return cubemap
