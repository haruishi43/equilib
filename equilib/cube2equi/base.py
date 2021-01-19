#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .cube2equi_numpy import run as run_numpy
from .cube2equi_torch import run as run_torch

__all__ = ["Cube2Equi", "cube2equi"]


class Cube2Equi(object):
    r"""
    params:
    - w_out, h_out (int): equirectangular image size
    - cube_format (str): input cube format("dice", "horizon", "dict", "list")
    - sampling_method (str): defaults to "default"
    - mode (str): interpolation mode, defaults to "bilinear"

    inputs:
    - cubemap (np.ndarray, torch.Tensor, dict, list)

    returns:
    - equi (np.ndarray, torch.Tensor)
    """

    def __init__(
        self,
        w_out: int,
        h_out: int,
        cube_format: str,
        sampling_method: str = "default",
        mode: str = "bilinear",
    ) -> None:
        assert w_out % 8 == 0 and h_out % 8 == 0
        self.w_out = w_out
        self.h_out = h_out
        self.cube_format = cube_format
        self.sampling_method = sampling_method
        self.mode = mode

    def __call__(
        self,
        cubemap: Union[
            np.ndarray,
            torch.Tensor,
            Dict[str, Union[np.ndarray, torch.Tensor]],
            List[Union[np.ndarray, torch.Tensor]],
        ],
    ) -> Union[np.ndarray, torch.Tensor]:
        return cube2equi(
            cubemap=cubemap,
            cube_format=self.cube_format,
            w_out=self.w_out,
            h_out=self.h_out,
            sampling_method=self.sampling_method,
            mode=self.mode,
        )


def cube2equi(
    cubemap: Union[
        np.ndarray,
        torch.Tensor,
        Dict[str, Union[np.ndarray, torch.Tensor]],
        List[Union[np.ndarray, torch.Tensor]],
    ],
    cube_format: str,
    w_out: int,
    h_out: int,
    sampling_method: str = "default",
    mode: str = "bilinear",
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    params:
    - cubemap: Union[
        np.ndarray,
        torch.Tensor,
        Dict[str, Union[np.ndarray, torch.Tensor]],
        List[Union[np.ndarray, torch.Tensor]]]
    - cube_format (str): ("dice", "horizon", "dict", "list")
    - w_out (int):
    - h_out (int):
    - sampling_method (str): "default"
    - mode (str): "bilinear"

    return:
    - equi (np.ndarray, torch.Tensor)
    """

    # Try and detect which type it is ("numpy" or "torch")
    # FIXME: any cleaner way of detecting?
    _type = None
    if cube_format in ("dice", "horizon"):
        if isinstance(cubemap, list):
            if isinstance(cubemap[0], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap[0], torch.Tensor):
                _type = "torch"
            else:
                raise ValueError
        else:
            if isinstance(cubemap, np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap, torch.Tensor):
                _type = "torch"
            else:
                raise ValueError
    elif cube_format == "dict":
        if isinstance(cubemap, dict):
            if isinstance(cubemap["F"], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap["F"], torch.Tensor):
                _type = "torch"
            else:
                raise ValueError
        elif isinstance(cubemap, list):
            if isinstance(cubemap[0]["F"], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap[0]["F"], torch.Tensor):
                _type = "torch"
            else:
                raise ValueError
        else:
            raise ValueError
    elif cube_format == "list":
        if isinstance(cubemap[0], list):
            if isinstance(cubemap[0][0], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap[0][0], torch.Tensor):
                _type = "torch"
            else:
                raise ValueError
        else:
            if isinstance(cubemap[0], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap[0], torch.Tensor):
                _type = "torch"
            else:
                raise ValueError
    else:
        raise ValueError

    if _type == "numpy":
        return run_numpy(
            cubemap=cubemap,
            cube_format=cube_format,
            w_out=w_out,
            h_out=h_out,
            sampling_method=sampling_method,
            mode=mode,
        )
    elif _type == "torch":
        return run_torch(
            cubemap=cubemap,
            cube_format=cube_format,
            w_out=w_out,
            h_out=h_out,
            sampling_method=sampling_method,
            mode=mode,
        )
    else:
        raise ValueError
