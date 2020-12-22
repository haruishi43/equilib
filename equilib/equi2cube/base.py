#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .equi2cube_numpy import run as run_numpy
from .equi2cube_torch import run as run_torch


__all__ = ["Equi2Cube", "equi2cube"]


class Equi2Cube(object):
    r"""Equi2Cube"""

    def __init__(
        self,
        w_face: int,
        cube_format: str,
        sampling_method: str = "default",
        mode: str = "bilinear",
    ) -> None:
        r"""
        params:
            w_face: cube face width (int)
        """
        self.w_face = w_face
        self.cube_format = cube_format
        self.sampling_method = sampling_method
        self.mode = mode

    def __call__(
        self,
        equi: Union[
            np.ndarray,
            torch.Tensor,
            List[Union[np.ndarray, torch.Tensor]],
        ],
        rot: Union[
            Dict[str, float],
            List[Dict[str, float]],
        ],
    ) -> Union[
        np.ndarray,
        torch.Tensor,
        Dict[str, Union[np.ndarray, torch.Tensor]],
        List[Union[np.ndarray, torch.Tensor]],
    ]:
        return equi2cube(
            equi=equi,
            rot=rot,
            w_face=self.w_face,
            cube_format=self.cube_format,
            sampling_method=self.sampling_method,
            mode=self.mode,
        )


def equi2cube(
    equi: Union[
        np.ndarray,
        torch.Tensor,
        List[Union[np.ndarray, torch.Tensor]],
    ],
    rot: Union[
        Dict[str, float],
        List[Dict[str, float]],
    ],
    w_face: int,
    cube_format: str,
    sampling_method: str = "default",
    mode: str = "bilinear",
) -> Union[
    np.ndarray,
    torch.Tensor,
    Dict[str, Union[np.ndarray, torch.Tensor]],
    List[Union[np.ndarray, torch.Tensor]],
]:

    # Try and detect which type it is ("numpy" or "torch")
    # FIXME: any cleaner way of detecting?
    _type = None
    if isinstance(equi, list):
        if isinstance(equi[0], np.ndarray):
            _type = "numpy"
        elif isinstance(equi[0], torch.Tensor):
            _type = "torch"
        else:
            raise ValueError
    elif isinstance(equi, np.ndarray):
        _type = "numpy"
    elif isinstance(equi, torch.Tensor):
        _type = "torch"
    else:
        raise ValueError

    if _type == "numpy":
        return run_numpy(
            equi=equi,
            rot=rot,
            w_face=w_face,
            cube_format=cube_format,
            sampling_method=sampling_method,
            mode=mode,
        )
    elif _type == "torch":
        return run_torch(
            equi=equi,
            rot=rot,
            w_face=w_face,
            cube_format=cube_format,
            sampling_method=sampling_method,
            mode=mode,
        )
    else:
        raise ValueError
