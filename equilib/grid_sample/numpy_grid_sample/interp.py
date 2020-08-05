#!/usr/bin/env python3


def linear_interp(v0, v1, d, l):
    r"""Basic Linear Interpolation
    """
    return v0*(1-d)/l + v1*d/l
