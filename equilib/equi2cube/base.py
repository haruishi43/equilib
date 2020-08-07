#!/usr/bin/env python3


class BaseEqui2Cube(object):
    r"""Base Equi2Cube class to build off of
    """

    def __init__(self, face_w: int):
        r"""
        params:
            face_w: cube face width (int)
        """
        self.face_w = face_w

    def __call__(self):
        raise NotImplementedError
