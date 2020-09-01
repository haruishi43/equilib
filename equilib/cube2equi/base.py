#!/usr/bin/env python3


class BaseCube2Equi(object):
    r"""Base Cube2Equi class to build off of"""

    def __init__(self, w_out: int, h_out: int, **kwargs) -> None:
        r"""
        params:
            w_out, h_out: equirectangular image size
        """
        assert w_out % 8 == 0 and h_out % 8 == 0
        self.w_out = w_out
        self.h_out = h_out

    def __call__(self, **kwargs):
        return self.run(**kwargs)

    def run(self, **kwargs):
        raise NotImplementedError
