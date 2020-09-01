#!/usr/bin/env python3

from typing import Optional


class BaseEqui2Equi(object):
    r"""Base Equi2Equi class to build off of"""

    def __init__(
        self, h_out: Optional[int] = None, w_out: Optional[int] = None, **kwargs
    ) -> None:
        r""""""
        self.h_out = h_out
        self.w_out = w_out

    def __call__(self, **kwargs):
        return self.run(**kwargs)

    def create_coordinate(self, h_out: int, w_out: int):
        raise NotImplementedError

    def rotation_matrix(self, roll: float, pitch: float, yaw: float):
        raise NotImplementedError

    @staticmethod
    def _get_img_size(img):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError
