#!/usr/bin/env python3

from typing import List, Tuple
import numpy as np


def create_coord(
    height: int,
    width: int,
) -> np.array:
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
) -> np.array:
    f = width / (2. * np.tan(np.radians(fov_x) / 2.))
    K = np.array([
        [f, 0., width/2],
        [0., f, height/2],
        [0., 0., 1.]])
    return K


def create_rot_mat(rot: List[float]) -> np.array:
    r"""param: rot: [yaw, pitch, roll]
    """
    rot_yaw, rot_pitch, rot_roll = rot

    R_yaw = np.array([
        [np.cos(rot_yaw), 0., -np.sin(rot_yaw)],
        [0., 1., 0.],
        [np.sin(rot_yaw), 0., np.cos(rot_yaw)]])
    R_pitch = np.array([
        [1., 0., 0.],
        [0., np.cos(rot_pitch), -np.sin(rot_pitch)],
        [0., np.sin(rot_pitch), np.cos(rot_pitch)]])
    R_roll = np.array([
        [np.cos(rot_roll), -np.sin(rot_roll), 0.],
        [np.sin(rot_roll), np.cos(rot_roll), 0.],
        [0., 0., 1.]])
    R = R_roll @ R_pitch @ R_yaw
    return R


def pixel_wise_rot(rot_coord: np.array) -> Tuple[np.array]:
    a = np.arctan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
    b = np.arcsin(rot_coord[:, :, 1] / np.linalg.norm(rot_coord, axis=2))
    return a, b
