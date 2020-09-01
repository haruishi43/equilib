#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

import numpy as np

from equilib.grid_sample import numpy_func

from ..base import BaseEqui2Pers
from .utils import create_rotation_matrix

__all__ = ["Equi2Pers"]


class Equi2Pers(BaseEqui2Pers):
    def __init__(self, **kwargs):
        r"""Equi2Pers Numpy"""
        super().__init__(**kwargs)

        # initialize intrinsic matrix
        _ = self.intrinsic_matrix
        # initialize global to camera rotation matrix
        _ = self.global2camera_rotation_matrix

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        r"""Create Intrinsic Matrix

        return:
            K: 3x3 matrix numpy.ndarray

        NOTE:
            ref: http://ksimek.github.io/2013/08/13/intrinsic/
        """
        if not hasattr(self, "_K"):
            # perspective projection (focal length)
            f = self.w_pers / (2.0 * np.tan(np.radians(self.fov_x) / 2.0))
            # transform between camera frame and pixel coordinates
            self._K = np.array(
                [
                    [f, self.skew, self.w_pers / 2],
                    [0.0, f, self.h_pers / 2],
                    [0.0, 0.0, 1.0],
                ]
            )
        return self._K

    @property
    def perspective_coordinate(self) -> np.ndarray:
        r"""Create mesh coordinate grid with perspective height and width

        return:
            coordinate: numpy.ndarray
        """
        _xs = np.linspace(0, self.w_pers - 1, self.w_pers)
        _ys = np.linspace(0, self.h_pers - 1, self.h_pers)
        xs, ys = np.meshgrid(_xs, _ys)
        zs = np.ones_like(xs)
        coord = np.stack((xs, ys, zs), axis=2)
        return coord

    @property
    def global2camera_rotation_matrix(self) -> np.ndarray:
        r"""Default rotation that changes global to camera coordinates"""
        if not hasattr(self, "_g2c_rot"):
            R_XY = np.array(
                [  # X <-> Y
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            R_YZ = np.array(
                [  # Y <-> Z
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            )
            self._g2c_rot = R_XY @ R_YZ

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

        Camera coordinates -> z-axis points forward, y-axis points upward
        Global coordinates -> x-axis points forward, z-axis points upward
        """
        R = create_rotation_matrix(x=roll, y=pitch, z=yaw)
        return R

    @staticmethod
    def _get_img_size(img: np.ndarray) -> Tuple[int]:
        r"""Return height and width"""
        return img.shape[-2:]

    def _run_single(
        self,
        equi: np.ndarray,
        rot: Dict[str, float],
        sampling_method: str,
        mode: str,
    ) -> np.ndarray:
        # define variables
        h_equi, w_equi = self._get_img_size(equi)
        m = self.perspective_coordinate
        K = self.intrinsic_matrix
        R = self.rotation_matrix(**rot)

        # conversion:
        K_inv = np.linalg.inv(K)
        m = m[:, :, :, np.newaxis]
        M = R @ self._g2c_rot @ K_inv @ m
        M = M.squeeze(3)

        # calculate rotations per perspective coordinates
        # phi = np.arcsin(M[:, :, 1] / np.linalg.norm(M, axis=-1))
        # theta = np.arctan2(M[:, :, 0], M[:, :, 2])
        phi = np.arcsin(M[:, :, 2] / np.linalg.norm(M, axis=-1))
        theta = np.arctan2(M[:, :, 1], M[:, :, 0])

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
        grid_sample = getattr(numpy_func, sampling_method, "faster")
        sampled = grid_sample(equi, grid, mode=mode)
        return sampled

    def run(
        self,
        equi: Union[np.ndarray, List[np.ndarray]],
        rot: Union[Dict[str, float], List[Dict[str, float]]],
        sampling_method: str = "faster",
        mode: str = "bilinear",
    ) -> np.ndarray:
        r"""Run Equi2Pers

        params:
            equi: equirectangular image np.ndarray[C, H, W]
            rot: Dict[str, float]
            sampling_method: str (default="faster")
            mode: str (default="bilinear")

        returns:
            pers: perspective image np.ndarray[C, H, W]

        NOTE: input can be batched [B, C, H, W] or List[np.ndarray]
        NOTE: when using batches, the output types match
        """
        _return_type = type(equi)
        _original_shape_len = len(equi.shape)
        if _return_type == np.ndarray:
            assert (
                _original_shape_len >= 3
            ), "ERR: got {} for input equi".format(_original_shape_len)
            if _original_shape_len == 3:
                equi = equi[np.newaxis, :, :, :]
                rot = [rot]

        assert len(equi) == len(
            rot
        ), "ERR: length of equi and rot differs {} vs {}".format(
            len(equi), len(rot)
        )

        samples = []
        for p, r in zip(equi, rot):
            # iterate through batches
            # TODO: batch implementation
            sample = self._run_single(
                equi=p,
                rot=r,
                sampling_method=sampling_method,
                mode=mode,
            )
            samples.append(sample)

        if _return_type == np.ndarray:
            samples = np.stack(samples, axis=0)
            if _original_shape_len == 3:
                samples = np.squeeze(samples, axis=0)

        return samples
