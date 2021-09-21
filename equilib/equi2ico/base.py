#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .numpy import (
    run as run_numpy,
)
from .torch import (
    run as run_torch,
)

__all__ = ["Equi2Ico", "equi2ico"]

ArrayLike = Union[np.ndarray, torch.Tensor]
SubLvl = Union[int, List[int]]
IcoMaps = Union[
    # single
    np.ndarray,
    # torch.Tensor,
    # single 'list'
    List[np.ndarray],
    List[torch.Tensor],
    # batch 'list'
    List[List[np.ndarray]],
    List[List[torch.Tensor]],
    # single 'dict'
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    # batch 'dict'
    List[Dict[str, np.ndarray]],
    List[Dict[str, np.ndarray]],
]

class Equi2Ico(object):
    """
    params:
    - w_face (int): icosahedron face width
    - fov_x (float): fov of horizontal axis in degrees
    - sub_level(int, list[int]): icosahedron subdivision level
    - ico_format (str): ("dict", "list")
    - mode (str)

    inputs:
    - equi (np.ndarray, torch.Tensor)

    returns:
    - ico_faces (np.ndarray, torch.Tensor, list, dict)
    """

    def __init__(
        self,
        w_face: int,
        fov_x: float,
        sub_level: SubLvl, 
        ico_format: str,
        mode: str = "bilinear",
    ) -> None:
        self.w_face = w_face
        self.fov_x = fov_x
        self.sub_level = sub_level
        self.ico_format = ico_format
        self.mode = mode

    def __call__(
        self,
        equi: ArrayLike,
    ) -> IcoMaps:
        return equi2ico(
            equi=equi,
            w_face=self.w_face,
            fov_x=self.fov_x,
            sub_level=self.sub_level,
            ico_format=self.ico_format,
            mode=self.mode,
        )

def equi2ico(
    equi: ArrayLike,
    w_face: int,
    fov_x: float,
    sub_level: SubLvl,
    ico_format: str,
    mode: str = "bilinear",
    **kwargs,
) -> IcoMaps:
    """
    params:
    - equi (np.ndarray, torch.Tensor)
    - w_face (int): icosahedron face width
    - fov_x (float): fov of horizontal axis in degrees
    - sub_level(int, list[int]): icosahedron subdivision level
    - ico_format (str): ("dict", "list")
    - mode (str)

    returns:
    - ico_faces (np.ndarray, torch.Tensor, dict, list)

    """

    _type = None
    if isinstance(equi, np.ndarray):
        _type = "numpy"
    elif torch.is_tensor(equi):
        _type = "torch"
    else:
        raise ValueError

    is_single = False
    if len(equi.shape) == 3 and isinstance(sub_level, int):
        # probably the input was a single image
        equi = equi[None, ...]
        sub_level = [sub_level]
        is_single = True
    elif len(equi.shape) == 3:
        # probably a grayscale image
        equi = equi[:, None, ...]

    assert isinstance(sub_level, list), "ERR: rots is not a list"
    if _type == "numpy":
        out = run_numpy(
            equi=equi,
            sub_level=sub_level,
            w_face=w_face,
            fov_x=fov_x,
            ico_format=ico_format,
            mode=mode,
            **kwargs,
        )
    elif _type == "torch":
        out = run_torch(
            equi=equi,
            sub_level=sub_level,
            w_face=w_face,
            fov_x=fov_x,
            ico_format=ico_format,
            mode=mode,
            **kwargs,
        )
    else:
        raise ValueError

    # make sure that the output batch dim is removed if it's only a single cubemap
    if is_single:
        out = out[0]

    return out