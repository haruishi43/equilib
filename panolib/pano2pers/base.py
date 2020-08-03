#!/usr/bin/env python3


class BasePano2Pers(object):
    r"""Base Pano2Pers clss to build off of
    """

    def __init__(
        self,
        w_pano: int,
        h_pano: int,
        w_pers: int,
        h_pers: int,
        fov_x: float,
        **kwargs,
    ) -> None:
        r"""
        params:
            w_pano, h_pano: panorama size
            w_pers, h_pers: perspective size
            fov_x: perspective image fov of x-axis (width direction)
        """
        self._init(
            w_pano=w_pano,
            h_pano=h_pano,
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
        )

    def _init_params(
        self,
        w_pano: int,
        h_pano: int,
        w_pers: int,
        h_pers: int,
        fov_x: float,
        skew: float = 0.,
    ) -> None:
        r"""Initialize local `self` parameters
        """
        self.w_pano = w_pano
        self.h_pano = h_pano
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

    def run(
        self,
        pano,
        rot,
        **kwargs,
    ):
        raise NotImplementedError

    def __call__(
        self,
        pano,
        rot,
        w_pers: int,
        h_pers: int,
        fov_x: float,
        **kwargs
    ):
        r"""Call Pano2Pers without initializing anything

        rot: Dict[roll, pitch, yaw] or List[Dict[roll, pitch, yaw]]
        """
        h_pano, w_pano = self._get_img_size(pano)
        self._init(
            w_pano=w_pano,
            h_pano=h_pano,
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
        )
        return self.run(
            pano=pano,
            rot=rot,
            **kwargs,
        )
