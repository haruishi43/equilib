#!/usr/bin/env python3


class BaseEqui2Cube(object):
    r"""Base Equi2Cube class to build off of"""

    def __init__(self, w_face: int, **kwargs):
        r"""
        params:
            w_face: cube face width (int)
        """
        self.w_face = w_face

    def __call__(self, **kwargs):
        return self.run(**kwargs)

    def run(self, **kwargs):
        raise NotImplementedError
