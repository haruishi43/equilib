#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

import numpy as np

from equilib.grid_sample import numpy_func

from .utils import create_rotation_matrix
from ..base import BaseEqui2Equi

__all__ = ["Equi2Equi"]


class Equi2Equi(BaseEqui2Equi):

    def __init__(self, **kwargs):
        r"""Equi2Equi Numpy
        """
        super().__init__(**kwargs)

        # initialize global to camera rotation matrix
        _ = self.global2camera_rotation_matrix

    @property
    def coordinate(self) -> np.ndarray:
        r"""Create mesh coordinate grid with height and width

        return:
            coordinate: numpy.ndarray
        """
        _xs = np.linspace(0, self.w_equi-1, self.w_equi)
        _ys = np.linspace(0, self.h_equi-1, self.h_equi)
        xs, ys = np.meshgrid(_xs, _ys)
        zs = np.ones_like(xs)
        coord = np.stack((xs, ys, zs), axis=2)
        return coord

    @property
    def global2camera_rotation_matrix(self) -> np.ndarray:
        r"""Default rotation that changes global to camera coordinates
        """
        if not hasattr(self, '_g2c_rot'):
            x = np.pi
            y = np.pi
            z = np.pi
            self._g2c_rot = create_rotation_matrix(x=x, y=y, z=z)
        return self._g2c_rot

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
        R_g2c = self.global2camera_rotation_matrix
        R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
        R = R_g2c @ R
        return R

    @staticmethod
    def _get_img_size(img: np.ndarray) -> Tuple[int]:
        r"""Return height and width"""
        return img.shape[-2:]

    def _run_single(
        self,
        src: np.ndarray,
        rot: Dict[str, float],
        sampling_method: str,
        mode: str,
    ) -> np.ndarray:
        # define variables
        h_equi, w_equi = self._get_img_size(src)

        assert h_equi == self.h_equi and w_equi == self.w_equi, \
            "ERR: size doesn't match for input and output"

        A = self.coordinate
        R = self.rotation_matrix(**rot)

        # conversion:
        # B = R @ A
        A = A[:, :, :, np.newaxis]
        B = R @ A
        B = B.squeeze(3)

        # calculate rotations per perspective coordinates
        phi = np.arcsin(B[:, :, 1] / np.linalg.norm(B, axis=-1))
        theta = np.arctan2(B[:, :, 0], B[:, :, 2])

        # center the image and convert to pixel location
        ui = (theta - np.pi) * w_equi / (2 * np.pi)
        uj = (phi - np.pi / 2) * h_equi / np.pi
        # out-of-bounds calculations
        ui = np.where(ui < 0, ui + w_equi, ui)
        ui = np.where(ui >= w_equi, ui - w_equi, ui)
        uj = np.where(uj < 0, uj + h_equi, uj)
        uj = np.where(uj >= h_equi, uj - h_equi, uj)
        grid = np.stack((uj, ui), axis=0)

        # grid sample
        grid_sample = getattr(
            numpy_func,
            sampling_method,
            "faster"
        )
        sampled = grid_sample(src, grid, mode=mode)
        return sampled

    def __call__(
        self,
        src: Union[np.ndarray, List[np.ndarray]],
        rot: Union[Dict[str, float], List[Dict[str, float]]],
        sampling_method: str = "faster",
        mode: str = "bilinear",
    ) -> np.ndarray:
        r"""Run Equi2Pers

        params:
            src: equirectangular image np.ndarray[C, H, W]
            rot: Dict[str, float]
            sampling_method: str (default="faster")
            mode: str (default="bilinear")

        returns:
            pers: perspective image np.ndarray[C, H, W]

        NOTE: input can be batched [B, C, H, W] or List[np.ndarray]
        NOTE: when using batches, the output types match
        """
        _return_type = type(src)
        _original_shape_len = len(src.shape)
        if _return_type == np.ndarray:
            assert _original_shape_len >= 3, \
                f"ERR: got {_original_shape_len} for input equi"
            if _original_shape_len == 3:
                src = src[np.newaxis, :, :, :]
                rot = [rot]

        assert len(src) == len(rot), \
            f"ERR: length of src and rot differs {len(src)} vs {len(rot)}"

        samples = []
        for s, r in zip(src, rot):
            # iterate through batches
            # TODO: batch implementation
            sample = self._run_single(
                src=s,
                rot=r,
                sampling_method=sampling_method,
                mode=mode,
            )
            samples.append(
                sample
            )

        if _return_type == np.ndarray:
            samples = np.stack(samples, axis=0)
            if _original_shape_len == 3:
                samples = np.squeeze(samples, axis=0)

        return samples
