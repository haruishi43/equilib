#!/usr/bin/env python3

from typing import List, Tuple
import numpy as np


def create_coord(
    height: int,
    width: int,
) -> np.ndarray:
    r"""Create mesh coordinate grid
    """
    _xs = np.linspace(0, width-1, width)
    _ys = np.linspace(0, height-1, height)
    xs, ys = np.meshgrid(_xs, _ys)
    zs = np.ones_like(xs)
    coord = np.stack((xs, ys, zs), axis=2)
    return coord


def create_K(
    height: int, width: int,
    fov_x: float,
) -> np.ndarray:
    r"""http://ksimek.github.io/2013/08/13/intrinsic/
    """
    # perspective projection (focal length)
    f = width / (2. * np.tan(np.radians(fov_x) / 2.))
    # axis skew
    s = 0.
    # transform between camera frame and pixel coordinates
    K = np.array([
        [f, s, width/2],
        [0., f, height/2],
        [0., 0., 1.]])
    return K


def create_rot_mat(rot: List[float]) -> np.ndarray:
    r"""param: rot: [roll, pitch, yaw] in radians
    """
    roll, pitch, yaw = rot
    
    # calculate rotation about the x-axis
    R_x = np.array([
        [1., 0., 0.],
        [0., np.cos(roll), -np.sin(roll)],
        [0., np.sin(roll), np.cos(roll)]])
    # calculate rotation about the y-axis
    R_y = np.array([
        [np.cos(pitch), 0., np.sin(pitch)],
        [0., 1., 0.],
        [-np.sin(pitch), 0., np.cos(pitch)]])
    # calculate rotation about the z-axis
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.],
        [np.sin(yaw), np.cos(yaw), 0.],
        [0., 0., 1.]])
    
    R = R_z @ R_y @ R_x
    return R


def pixel_wise_rot(rot_coord: np.ndarray) -> Tuple[np.ndarray]:
    phis = np.arcsin(rot_coord[:, :, 1] / np.linalg.norm(rot_coord, axis=2))
    thetas = np.arctan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
    return phis, thetas
