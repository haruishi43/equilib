#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .numpy import (
    run as run_numpy
)
from .torch import (
    run as run_torch,
)

__all__ = ["Ico2Equi", "ico2equi"]

ArrayLike = Union[np.ndarray, torch.Tensor]
IcoMaps = Union[
    # single/batch 'horizon' or 'dice'
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


class Ico2Equi(object):
    """
    params:
    - height, width (int): equirectangular image size
    - ico_format (str): input cube format("dict", "list")
    - mode (str): interpolation mode, defaults to "bilinear"

    inputs:
    - icomap (dict, list)

    returns:
    - equi (np.ndarray, torch.Tensor)
    """

    def __init__(
        self,
        height: int,
        width: int,
        fov_x: float,
        ico_format: str,
        mode: str = "bilinear",
    ) -> None:
        self.height = height
        self.width = width
        self.fov_x = fov_x
        self.ico_format = ico_format
        self.mode = mode

    def __call__(
        self,
        icomap: IcoMaps,
        **kwargs,
    ) -> ArrayLike:
        return ico2equi(
            icomap=icomap,
            ico_format=self.ico_format,
            width=self.width,
            height=self.height,
            fov_x=self.fov_x,
            mode=self.mode,
            **kwargs,
        )


def ico2equi(
    icomap: IcoMaps,
    ico_format: str,
    height: int,
    width: int,
    fov_x: float,
    mode: str = "bilinear",
    **kwargs,
) -> ArrayLike:
    """
    params:
    - icomaps
    - ico_format (str): ("dict", "list")
    - height, width (int): output size
    - mode (str): "bilinear"

    return:
    - equi (np.ndarray, torch.Tensor)
    """

    assert width % 8 == 0 and height % 8 == 0

    # Try and detect which type it is ("numpy" or "torch")
    # FIXME: any cleaner way of detecting?
    _type = None
    if ico_format == "dict":
        if isinstance(icomap, dict):
            if isinstance(icomap["0"], np.ndarray):
                _type = "numpy"
            elif isinstance(icomap["0"], torch.Tensor):
                _type = "torch"
        elif isinstance(icomap, list):
            assert isinstance(icomap[0], dict)
            if isinstance(icomap[0]["0"], np.ndarray):
                _type = "numpy"
            elif isinstance(icomap[0]["0"], torch.Tensor):
                _type = "torch"
    elif ico_format == "list":
        assert isinstance(icomap, list)
        if isinstance(icomap[0], list):
            if isinstance(icomap[0][0], np.ndarray):
                _type = "numpy"
            elif isinstance(icomap[0][0], torch.Tensor):
                _type = "torch"
        else:
            if isinstance(icomap[0], np.ndarray):
                _type = "numpy"
            elif isinstance(icomap[0], torch.Tensor):
                _type = "torch"
    assert _type is not None, "ERR: input type is not numpy or torch"
    if _type == "numpy":
        #TODO: add in future
        #icomaps_batch = convert2batches(icomap, ico_format)
        out = run_numpy(
            icomaps=icomap,
            height=height,
            width=width,
            fov_x=fov_x,
            mode=mode,
            **kwargs,
        )
    elif _type == "torch":
        out = run_torch(
            icomaps=icomap,
            height=height,
            width=width,
            fov_x=fov_x,
            mode=mode,
            **kwargs,
        )
    else:
        raise ValueError("Oops something went wrong here")

    if out.shape[0] == 1:
        out = out.squeeze(0)

    return out
