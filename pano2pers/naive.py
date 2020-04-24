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
    f = width / (2 * np.tan(np.radians(fov_x) / 2))
    K = np.array([
        [f, 0, width/2],
        [0, f, height/2],
        [0, 0, 1]])
    return K


def create_rot_mat(rot: List[float]) -> np.array:
    r"""param: rot: [yaw, pitch, roll]
    """
    rot_yaw, rot_pitch, rot_roll = rot

    R_yaw = np.array([
        [np.cos(rot_yaw), 0, -np.sin(rot_yaw)],
        [0, 1, 0],
        [np.sin(rot_yaw), 0, np.cos(rot_yaw)]])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(rot_pitch), -np.sin(rot_pitch)],
        [0, np.sin(rot_pitch), np.cos(rot_pitch)]])
    R_roll = np.array([
        [np.cos(rot_roll), -np.sin(rot_roll), 0],
        [np.sin(rot_roll), np.cos(rot_roll), 0],
        [0, 0, 1]])
    R = R_roll @ R_pitch @ R_yaw
    return R


def pixel_wise_rot(rot_coord: np.array) -> Tuple[np.array]:
    a = np.arctan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
    b = np.arcsin(rot_coord[:, :, 1] / np.linalg.norm(rot_coord, axis=2))
    return a, b


def interp3d(
    Q: List[np.array],
    dy: float, dx: float,
    mode: str = 'bilinear',
) -> np.array:
    r"""Naive Interpolation
        (y,x): target pixel
        mode: interpolation mode
    """
    q_00, q_10, q_01, q_11 = Q
    if mode == 'bilinear':
        f_0 = q_00*dx + q_01*(1-dx)
        f_1 = q_10*dx + q_11*(1-dx)
        f_3 = f_0*dy + f_1*(1-dy)
        out = f_3
    else:
        print(f"{mode} is not supported")
    return out


def grid_sample(
    img: np.array, grid: np.array,
    mode: str = 'bilinear',
) -> np.array:
    r"""
    """
    channels, h_in, w_in = img.shape
    if img.dtype == np.uint8:
        _min = 0
        _max = 255
        _dtype = np.uint8
    elif img.dtype == np.float64:
        _min = 0.0
        _max = 1.0
        _dtype = np.float64
    else:
        print(f"{img.dtype} is not supported")
    
    _, h_out, w_out = grid.shape
    out = np.zeros((channels, h_out, w_out), dtype=_dtype)

    for y in range(h_out):
        for x in range(w_out):
            y_in, x_in = grid[:,y,x]
            
            y0 = int(np.floor(y_in))
            y1 = int(np.floor(y_in)) + 1
            x0 = int(np.floor(x_in))
            x1 = int(np.floor(x_in)) + 1

            dy = y_in - y0
            dx = x_in - x0

            if y1 >= h_in:
                y1 = y1 - h_in
            if x1 >= w_in:
                x1 = x1 - w_in

            q_00 = img[:, y0, x0]
            q_10 = img[:, y1, x0]
            q_01 = img[:, y0, x1]
            q_11 = img[:, y1, x1]

            f_0 = q_00*(x1-x_in) + q_01*(x_in-x0)
            f_1 = q_10*(x1-x_in) + q_11*(x_in-x0)
            f_3 = f_0*(y1-y_in) + f_1*(y_in-y0)
            _out = f_3
            _out = np.where(_out >= _max, _max, _out)
            _out = np.where(_out < _min, _min, _out)
            out[:,y,x] = _out.astype(_dtype)
    return out


if __name__ == "__main__":
    import os
    import os.path as osp

    from PIL import Image

    data_path = osp.join('..', 'data')
    pano_path = osp.join(data_path, 'pano2.png')

    pano_img = Image.open(pano_path)

    # Sometimes images are RGBA
    pano_img = pano_img.convert('RGB')
    pano = np.asarray(pano_img)

    pano = np.transpose(pano, (2,0,1))
    _, h_pano, w_pano = pano.shape
    print('panorama size:')
    print(h_pano, w_pano)
    
    # Variables:
    h_pers = 480
    w_pers = 640
    rot = [0, 0, 0]
    fov_x = 80

    coord = create_coord(h_pers, w_pers)
    K = create_K(h_pers, w_pers, fov_x)
    R = create_rot_mat(rot)

    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)
    coord = coord[:, :, :, np.newaxis]

    rot_coord = R_inv @ K_inv @ coord
    rot_coord = rot_coord.squeeze(3)

    a, b = pixel_wise_rot(rot_coord)

    ui = (a + np.pi) * w_pano / (2 * np.pi)
    uj = (b + np.pi / 2) * h_pano / np.pi

    ui = np.where(ui < 0, ui + w_pano, ui)
    ui = np.where(ui >= w_pano, ui - w_pano, ui)
    uj = np.where(uj < 0, uj + h_pano, uj)
    uj = np.where(uj >= h_pano, uj - h_pano, uj)

    # pano = pano / 255.

    grid = np.stack((uj, ui), axis=0)
    sampled = grid_sample(pano, grid)

    # after sample
    pers = np.transpose(sampled, (1,2,0))
    # pers = (pers * 255).astype(np.uint8)
    pers_img = Image.fromarray(pers)

    pers_path = osp.join(data_path, 'output.jpg')
    pers_img.save(pers_path)