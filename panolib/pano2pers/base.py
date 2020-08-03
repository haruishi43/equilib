#!/usr/bin/env python3


class BasePano2Pers(object):
    r"""Base Pano2Pers clss to build off of
    """

    def __init__(
        self,
        w_pers: int,
        h_pers: int,
        fov_x: float,
        **kwargs,
    ) -> None:
        r"""
        params:
            w_pers, h_pers: perspective size
            fov_x: perspective image fov of x-axis (width direction)
        """
        self._init_params(
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
        )

    def _init_params(
        self,
        w_pers: int,
        h_pers: int,
        fov_x: float,
        skew: float = 0.,
    ) -> None:
        r"""Initialize local `self` parameters
        """
        self.w_pers = w_pers
        self.h_pers = h_pers
        self.fov_x = fov_x
        self.skew = skew  # skew intrinsic parameter

    @property
    def intrinsic_matrix(self):
        raise NotImplementedError

    @property
    def perspective_coordinate(self):
        raise NotImplementedError

    @property
    def global2camera_rotation_matrix(self):
        raise NotImplementedError

    def rotation_matrix(self, roll: float, pitch: float, yaw: float):
        raise NotImplementedError

    @staticmethod
    def _get_img_size(img):
        raise NotImplementedError

    def __call__(
        self,
        pano,
        rot,
        **kwargs,
    ):
        raise NotImplementedError
