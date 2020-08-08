#!/usr/bin/env python3

from typing import Dict, List, Union

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
    create_rotation_matrix,
)
from ..base import BaseEqui2Cube


class Equi2Cube(BaseEqui2Cube):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def create_xyz(self, w_face: int):
        r"""xyz coordinates of the faces of the cube
        """
        out = np.zeros((w_face, w_face * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=w_face, dtype=np.float32)
        grid = np.stack(np.meshgrid(-rng, rng), -1)

        # Front face (x = 0.5)
        out[:, 0*w_face:1*w_face, [1, 2]] = np.stack(
            np.meshgrid(rng, -rng), -1
        )
        out[:, 0*w_face:1*w_face, 0] = 0.5

        # Right face (y = -0.5)
        out[:, 1*w_face:2*w_face, [0, 2]] = np.stack(
            np.meshgrid(-rng, -rng), -1
        )
        out[:, 1*w_face:2*w_face, 1] = 0.5

        # Back face (x = -0.5)
        out[:, 2*w_face:3*w_face, [1, 2]] = np.stack(
            np.meshgrid(-rng, -rng), -1
        )
        out[:, 2*w_face:3*w_face, 0] = -0.5

        # Left face (y = 0.5)
        out[:, 3*w_face:4*w_face, [0, 2]] = np.stack(
            np.meshgrid(rng, -rng), -1
        )
        out[:, 3*w_face:4*w_face, 1] = -0.5

        # Up face (z = 0.5)
        out[:, 4*w_face:5*w_face, [1, 0]] = np.stack(
            np.meshgrid(rng, rng), -1
        )
        out[:, 4*w_face:5*w_face, 2] = 0.5

        # Down face (z = -0.5)
        out[:, 5*w_face:6*w_face, [1, 0]] = np.stack(
            np.meshgrid(rng, -rng), -1
        )
        out[:, 5*w_face:6*w_face, 2] = -0.5

        return out

    def xyz2rot(self, xyz):
        r"""Return rotation (theta, phi) from xyz
        """
        norm = np.linalg.norm(xyz, axis=-1)
        phi = np.arcsin(xyz[:, :, 2] / norm)
        theta = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
        return theta, phi

    def __call__(
        self,
        equi: Union[np.ndarray, List[np.ndarray]],
        rot: Union[Dict[str, float], List[Dict[str, float]]],
        cube_format: str,
        sampling_method: str = 'faster',
        mode: str = "bilinear",
    ) -> Union[np.ndarray, List[np.ndarray]]:
        r"""Call Equi2Cube

        params:
            equi: Union[np.ndarray, List[np.ndarray]]
            rot: Union[Dict[str, float], List[Dict[str, float]]]
            cube_format: str ('list', 'dict', 'dice')
            sampling_method: str (default = 'faster')
            mode: str (default = 'bilinear')
        """

        assert len(equi.shape) == 3, f"ERR: {equi.shape} is not a valid array"
        assert equi.shape[0] == 3, f"ERR: got {equi.shape[0]} for channel size"
        h_equi, w_equi = equi.shape[-2:]

        xyz = self.create_xyz(self.w_face)
        xyz = xyz[:, :, :, np.newaxis]
        xyz_ = self.rotation_matrix(**rot) @ xyz
        xyz_ = xyz_.squeeze(-1)
        theta, phi = self.xyz2rot(xyz_)

        # center the image and convert to pixel location
        ui = (theta - np.pi) * w_equi / (2 * np.pi)
        uj = (np.pi / 2 - phi) * h_equi / np.pi

        # out-of-bounds calculations
        ui = np.where(ui < 0, ui + w_equi, ui)
        ui = np.where(ui >= w_equi, ui - w_equi, ui)
        uj = np.where(uj < 0, uj + h_equi, uj)
        uj = np.where(uj >= h_equi, uj - h_equi, uj)
        grid = np.stack((uj, ui), axis=0)

        grid_sample = getattr(
            numpy_func,
            sampling_method,
            'faster'
        )
        cubemap = grid_sample(equi, grid, mode=mode)

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
