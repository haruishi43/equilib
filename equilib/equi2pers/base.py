#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .equi2pers_numpy import run as run_numpy
from .equi2pers_torch import run as run_torch

__all__ = ["Equi2Pers", "equi2pers"]


class Equi2Pers(object):
    """
    params:
    - w_pers, h_pers (int): perspective size
    - fov_x (float): perspective image fov of x-axis
    - skew (float): skew intrinsic parameter
    - sampling_method (str)
    - mode (str)
    - z_down (bool)

    inputs:
    - equi (np.ndarray, torch.Tensor, list)
    - rot (dict, list): Dict[str, float]

    returns:
    - pers (np.ndarray, torch.Tensor)
    """

    def __init__(
        self,
        w_pers: int,
        h_pers: int,
        fov_x: float,
        skew: float = 0.0,
        sampling_method: str = "default",
        mode: str = "bilinear",
        z_down: bool = False,
    ) -> None:
        self.w_pers = w_pers
        self.h_pers = h_pers
        self.fov_x = fov_x
        self.skew = skew
        self.sampling_method = sampling_method
        self.mode = mode
        self.z_down = z_down

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
    ) -> Union[np.ndarray, torch.Tensor]:
        return equi2pers(
            equi=equi,
            rot=rot,
            w_pers=self.w_pers,
            h_pers=self.h_pers,
            fov_x=self.fov_x,
            skew=self.skew,
            sampling_method=self.sampling_method,
            mode=self.mode,
            z_down=self.z_down,
        )


def equi2pers(
    equi: Union[
        np.ndarray,
        torch.Tensor,
        List[Union[np.ndarray]],
    ],
    rot: Union[
        Dict[str, float],
        List[Dict[str, float]],
    ],
    w_pers: int,
    h_pers: int,
    fov_x: float,
    skew: float = 0.0,
    sampling_method: str = "default",
    mode: str = "bilinear",
    z_down: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    params:
    - equi (np.ndarray, torch.Tensor, list)
    - rot (dict, list): Dict[str, float]
    - w_pers, h_pers (int): perspective size
    - fov_x (float): perspective image fov of x-axis
    - skew (float): skew intrinsic parameter
    - z_down (bool)

    returns:
    - pers (np.ndarray, torch.Tensor)
    """

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
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
            skew=skew,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )
    elif _type == "torch":
        return run_torch(
            equi=equi,
            rot=rot,
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
            skew=skew,
            sampling_method=sampling_method,
            mode=mode,
            z_down=z_down,
        )
    else:
        raise ValueError
