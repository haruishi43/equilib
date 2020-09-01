#!/usr/bin/env python3

import math
from typing import Dict, List, Union

import torch

from equilib.grid_sample import torch_func

from ..base import BaseEqui2Cube
from .utils import (
    create_rotation_matrix,
    cube_h2dice,
    cube_h2dict,
    cube_h2list,
    get_device,
    sizeof,
)


class Equi2Cube(BaseEqui2Cube):
    r"""Equi2Cube PyTorch"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rotation_matrix(
        self,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> torch.Tensor:
        r"""Create Rotation Matrix

        params:
            roll: x-axis rotation float
            pitch: y-axis rotation float
            yaw: z-axis rotation float

        return:
            rotation matrix: torch.Tensor
        """
        R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
        return R

    def create_xyz(self, w_face: int):
        r"""xyz coordinates of the faces of the cube"""
        _dtype = torch.float32
        out = torch.zeros((w_face, w_face * 6, 3), dtype=_dtype)
        rng = torch.linspace(-0.5, 0.5, w_face, dtype=_dtype)

        # NOTE: https://github.com/pytorch/pytorch/issues/15301
        # Torch meshgrid behaves differently than numpy

        # Front face (x = 0.5)
        out[:, 0 * w_face : 1 * w_face, [2, 1]] = torch.stack(
            torch.meshgrid([-rng, rng]), -1
        )
        out[:, 0 * w_face : 1 * w_face, 0] = 0.5

        # Right face (y = -0.5)
        out[:, 1 * w_face : 2 * w_face, [2, 0]] = torch.stack(
            torch.meshgrid([-rng, -rng]), -1
        )
        out[:, 1 * w_face : 2 * w_face, 1] = 0.5

        # Back face (x = -0.5)
        out[:, 2 * w_face : 3 * w_face, [2, 1]] = torch.stack(
            torch.meshgrid([-rng, -rng]), -1
        )
        out[:, 2 * w_face : 3 * w_face, 0] = -0.5

        # Left face (y = 0.5)
        out[:, 3 * w_face : 4 * w_face, [2, 0]] = torch.stack(
            torch.meshgrid([-rng, rng]), -1
        )
        out[:, 3 * w_face : 4 * w_face, 1] = -0.5

        # Up face (z = 0.5)
        out[:, 4 * w_face : 5 * w_face, [0, 1]] = torch.stack(
            torch.meshgrid([rng, rng]), -1
        )
        out[:, 4 * w_face : 5 * w_face, 2] = 0.5

        # Down face (z = -0.5)
        out[:, 5 * w_face : 6 * w_face, [0, 1]] = torch.stack(
            torch.meshgrid([-rng, rng]), -1
        )
        out[:, 5 * w_face : 6 * w_face, 2] = -0.5

        return out

    def xyz2rot(self, xyz):
        r"""Return rotation (theta, phi) from xyz"""
        norm = torch.norm(xyz, dim=-1)
        phi = torch.asin(xyz[:, :, :, 2] / norm)
        theta = torch.atan2(xyz[:, :, :, 1], xyz[:, :, :, 0])
        return theta, phi

    def run(
        self,
        equi: Union[torch.Tensor, List[torch.Tensor]],
        rot: Union[Dict[str, float], List[Dict[str, float]]],
        cube_format: str,
        sampling_method: str = "torch",
        mode: str = "bilinear",
        debug: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor], List[dict]]:
        r"""Call Equi2Cube

        params:
            equi: Union[torch.Tensor, List[torch.Tensor]]
            rot: Union[Dict[str, float], List[Dict[str, float]]]
            cube_format: str ('list', 'dict', 'dice')
            sampling_method: str (default = 'faster')
            mode: str (default = 'bilinear')
        """

        assert type(equi) == torch.Tensor, (
            "ERR: input equi expected to be `torch.Tensor` "
            "but got {}".format(type(equi))
        )
        _original_shape_len = len(equi.shape)
        assert _original_shape_len >= 3, "ERR: got {} for input equi".format(
            _original_shape_len
        )
        if _original_shape_len == 3:
            equi = equi.unsqueeze(dim=0)
            rot = [rot]

        h_equi, w_equi = equi.shape[-2:]
        if debug:
            print("equi: ", sizeof(equi) / 10e6, "mb")

        # get device
        device = get_device(equi)

        # define variables
        xyz = []
        for r in rot:
            # for each rotations calculate M
            _xyz = self.create_xyz(self.w_face)
            xyz_ = self.rotation_matrix(**r) @ _xyz.unsqueeze(3)
            xyz_ = xyz_.squeeze(3)
            xyz.append(xyz_)
        xyz = torch.stack(xyz, dim=0).to(device)

        theta, phi = self.xyz2rot(xyz)

        # center the image and convert to pixel location
        ui = (theta - math.pi) * w_equi / (2 * math.pi)
        uj = (math.pi / 2 - phi) * h_equi / math.pi

        # out-of-bounds calculations
        ui = torch.where(ui < 0, ui + w_equi, ui)
        ui = torch.where(ui >= w_equi, ui - w_equi, ui)
        uj = torch.where(uj < 0, uj + h_equi, uj)
        uj = torch.where(uj >= h_equi, uj - h_equi, uj)
        grid = torch.stack((uj, ui), dim=-3)

        grid_sample = getattr(
            torch_func,
            sampling_method,
        )
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
            raise NotImplementedError()

        return cubemap
