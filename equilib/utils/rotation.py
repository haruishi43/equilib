#!/usr/bin/env python3

from typing import Union

import numpy as np

import torch

from .numpy_utils.rotation import (
    create_global2camera_rotation_matrix as np_grm,
    create_intrinsic_matrix as np_im,
    create_rotation_matrix as np_rm,
)
from .torch_utils.rotation import (
    create_global2camera_rotation_matrix as th_grm,
    create_intrinsic_matrix as th_im,
    create_rotation_matrix as th_rm,
)


def create_global2camera_rotation_matrix(
    is_torch: bool,
) -> Union[np.ndarray, torch.Tensor]:
    """Create Rotation Matrix that transform from global to camera coordinate

    params:
    - is_torch (bool): return torch.Tensor

    return:
    - rotation matrix (np.ndarray, torch.Tensor): 3x3 rotation matrix
    """
    if is_torch:
        return th_grm()
    else:
        return np_grm()


def create_intrinsic_matrix(
    w_pers: int,
    h_pers: int,
    fov_x: float,
    skew: float,
    is_torch: bool,
) -> Union[np.ndarray, torch.Tensor]:
    """Create Intrinsic Matrix

    params:
    - w_pers (int)
    - h_pers (int)
    - fov_x (float)
    - skew (float)
    - is_torch (bool): return torch.Tensor

    return:
    - K (np.ndarray, torch.Tensor): 3x3 matrix

    NOTE:
    - ref: http://ksimek.github.io/2013/08/13/intrinsic/
    """
    if is_torch:
        return th_im(
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
            skew=skew,
        )
    else:
        return np_im(
            w_pers=w_pers,
            h_pers=h_pers,
            fov_x=fov_x,
            skew=skew,
        )


def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    z_down: bool,
    is_torch: bool,
) -> Union[np.ndarray, torch.Tensor]:
    """Create Rotation Matrix

    params:
    - roll (float): x-axis rotation
    - pitch (float): y-axis rotation
    - yaw (float): z-axis rotation
    - z_down (bool): make z-axis face down
    - is_torch (bool): return torch.Tensor

    return:
    - rotation matrix (np.ndarray, torch.Tensor): 3x3 matrix
    """
    if is_torch:
        return th_rm(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            z_down=z_down,
        )
    else:
        return np_rm(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            z_down=z_down,
        )
