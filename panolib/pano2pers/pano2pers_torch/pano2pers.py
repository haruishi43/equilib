#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

import torch

from panolib.grid_sample import torch_func

from .utils import (
    create_rotation_matrix,
    deg2rad,
    get_device,
    pi,
)
from ..base import BasePano2Pers

__all___ = ["Pano2Pers"]


class Pano2Pers(BasePano2Pers):

    def __init__(self, **kwargs):
        r"""Pano2Pers PyTorch
        """
        super().__init__(**kwargs)

        # initialize intrinsic matrix
        _ = self.intrinsic_matrix
        # initialize global to camera rotation matrix
        _ = self.global2camera_rotation_matrix

    @property
    def intrinsic_matrix(self) -> torch.Tensor:
        r"""Create Intrinsic Matrix

        return:
            K: 3x3 matrix torch.Tensor

        NOTE:
            ref: http://ksimek.github.io/2013/08/13/intrinsic/
        """
        if not hasattr(self, '_K'):
            fov_x = torch.tensor(self.fov_x)
            f = self.w_pers / (2 * torch.tan(deg2rad(fov_x) / 2))
            self._K = torch.tensor([
                [f, self.skew, self.w_pers/2],
                [0., f, self.h_pers/2],
                [0., 0., 1.]])
        return self._K

    @property
    def perspective_coordinate(self) -> torch.Tensor:
        r"""Create mesh coordinate grid with perspective height and width

        return:
            coordinate: torch.Tensor
        """
        _xs = torch.linspace(0, self.w_pers-1, self.w_pers)
        _ys = torch.linspace(0, self.h_pers-1, self.h_pers)
        # NOTE: https://github.com/pytorch/pytorch/issues/15301
        # Torch meshgrid behaves differently than numpy
        ys, xs = torch.meshgrid([_ys, _xs])
        zs = torch.ones_like(xs)
        coord = torch.stack((xs, ys, zs), dim=2)
        return coord

    @property
    def global2camera_rotation_matrix(self) -> torch.Tensor:
        r"""Default rotation that changes global to camera coordinates
        """
        if not hasattr(self, '_g2c_rot'):
            x = pi
            y = pi
            z = pi
            self._g2c_rot = create_rotation_matrix(x=x, y=y, z=z)
        return self._g2c_rot

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

        Camera coordinates -> z-axis points forward, y-axis points upward
        Global coordinates -> x-axis points forward, z-axis poitns upward
        """
        R_g2c = self.global2camera_rotation_matrix
        R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
        R = R_g2c @ R
        return R

    @staticmethod
    def _get_img_size(img: torch.Tensor) -> Tuple[int]:
        r"""Return height and width"""
        # batch, channel, height, width
        return img.shape[-2], img.shape[-1]

    def __call__(
        self,
        pano: torch.Tensor,
        rot: Union[Dict[str, float], List[Dict[str, float]]],
        sampling_method: str = "torch",
        mode: str = "bilinear",
    ) -> torch.Tensor:
        r"""Run Pano2Pers

        params:
            pano: panorama image torch.Tensor[(B), C, H, W]
            rot: Dict[str, float] or List[Dict[str, float]]
            sampling_method: str (default="torch")
            mode: str (default="bilinear")

        returns:
            pers: perspective image torch.Tensor[C, H, W]

        NOTE: input can be batched [B, C, H, W] or single [C, H, W]
        NOTE: when using batches, the output types match
        """
        assert type(pano) == torch.Tensor, \
            (
                "ERR: input pano expected to be `torch.Tensor` "
                f"but got {type(pano)}"
            )
        _original_shape_len = len(pano.shape)
        assert _original_shape_len >= 3, \
            f"ERR: got {_original_shape_len} for input pano"
        if _original_shape_len == 3:
            pano = pano.unsqueeze(dim=0)
            rot = [rot]

        # get device
        device = get_device(pano)

        # define variables
        M = []
        for r in rot:
            # for each rotations calculate M
            m = self.perspective_coordinate
            K = self.intrinsic_matrix
            R = self.rotation_matrix(**r)
            _M = R.inverse() @ K.inverse() @ m.unsqueeze(3)
            _M = _M.squeeze(3)
            M.append(_M)
        M = torch.stack(M, dim=0).to(device)

        # calculate rotations per perspective coordinates
        norms = torch.norm(M, dim=-1)
        theta = torch.atan2(M[:, :, :, 0], M[:, :, :, 2])
        phi = torch.asin(M[:, :, :, 1] / norms)

        # center the image and convert to pixel locatio
        ui = (theta - pi) * self.w_pano / (2 * pi)
        uj = (phi - pi / 2) * self.h_pano / pi
        # out-of-bounds calculations
        ui = torch.where(ui < 0, ui + self.w_pano, ui)
        ui = torch.where(ui >= self.w_pano, ui - self.w_pano, ui)
        uj = torch.where(uj < 0, uj + self.h_pano, uj)
        uj = torch.where(uj >= self.h_pano, uj - self.h_pano, uj)
        grid = torch.stack((uj, ui), axis=-3)  # 3rd to last

        # grid sample
        grid_sample = getattr(
            torch_func,
            sampling_method,
            "torch"
        )
        samples = grid_sample(pano, grid, mode=mode)

        if _original_shape_len == 3:
            samples = samples.squeeze(axis=0)

        return samples
