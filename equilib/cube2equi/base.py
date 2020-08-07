#!/usr/bin/env python3


class BaseCube2Equi(object):
    r"""Base Cube2Equi class to build off of
    """

    def __init__(self, w_face: int, **kwargs):
        self.w_face = w_face

    def __call__(self):
        raise NotImplementedError
