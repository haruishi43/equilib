#!/usr/bin/env python3


class BaseEqui2Equi(object):
    r"""Base Equi2Equi class to build off of
    """

    def __init__(
        self,
        h_equi: int,
        w_equi: int,
        **kwargs
    ) -> None:
        r"""
        """
        self.h_equi = h_equi
        self.w_equi = w_equi

    @property
    def coordinate(self):
        raise NotImplementedError

    @property
    def global2camera_rotation_matrix(self):
        raise NotImplementedError

    def rotation_matrix(self, roll: float, pitch: float, yaw: float):
        raise NotImplementedError

    @staticmethod
    def _get_img_size(img):
        raise NotImplementedError

    def __call__(self, src, rot, **kwargs):
        raise NotImplementedError
