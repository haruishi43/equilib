#!/usr/bin/env python3

from typing import Dict, List, Optional, Union

import numpy as np

import torch

from .equi2equi_numpy import run as run_numpy
from .equi2equi_torch import run as run_torch

__all__ = ["Equi2Equi", "equi2equi"]


class Equi2Equi(object):
    r"""
    init params:
    - w_out, h_out (optional int): equi image size
    - sampling_method (str): defaults to "default"
    - mode (str): interpolation mode, defaults to "bilinear"

    input params:
    - src (np.ndarray, torch.Tensor, list)
    - rot (dict, list[dict])

    return:
    - equi (np.ndarray, torch.Tensor)
    """

    def __init__(
        self,
        w_out: Optional[int] = None,
        h_out: Optional[int] = None,
        sampling_method: str = "default",
        mode: str = "bilinear",
    ) -> None:
        self.w_out = w_out
        self.h_out = h_out
        self.sampling_method = sampling_method
        self.mode = mode

    def __call__(
        self,
        src: Union[
            np.ndarray,
            torch.Tensor,
            List[Union[np.ndarray, torch.Tensor]],
        ],
        rot: Union[
            Dict[str, float],
            List[Dict[str, float]],
        ],
    ) -> Union[np.ndarray, torch.Tensor]:
        return equi2equi(
            src=src,
            rot=rot,
            sampling_method=self.sampling_method,
            mode=self.mode,
        )


def equi2equi(
    src: Union[
        np.ndarray,
        torch.Tensor,
        List[Union[np.ndarray, torch.Tensor]],
    ],
    rot: Union[
        Dict[str, float],
        List[Dict[str, float]],
    ],
    sampling_method: str = "default",
    mode: str = "bilinear",
    w_out: Optional[int] = None,
    h_out: Optional[int] = None,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    init params:
    - src (np.ndarray, torch.Tensor, list)
    - rot (dict, list[dict])
    - w_out, h_out (optional int): equi image size
    - sampling_method (str): defaults to "default"
    - mode (str): interpolation mode, defaults to "bilinear"
    """

    # Try and detect which type it is ("numpy" or "torch")
    # FIXME: any cleaner way of detecting?
    _type = None
    if isinstance(src, list):
        if isinstance(src[0], np.ndarray):
            _type = "numpy"
        elif isinstance(src[0], torch.Tensor):
            _type = "torch"
        else:
            raise ValueError
    elif isinstance(src, np.ndarray):
        _type = "numpy"
    elif isinstance(src, torch.Tensor):
        _type = "torch"
    else:
        raise ValueError

    if _type == "numpy":
        return run_numpy(
            src=src,
            rot=rot,
            sampling_method=sampling_method,
            mode=mode,
            w_out=w_out,
            h_out=h_out,
        )
    elif _type == "torch":
        return run_torch(
            src=src,
            rot=rot,
            sampling_method=sampling_method,
            mode=mode,
            w_out=w_out,
            h_out=h_out,
        )
    else:
        raise ValueError
