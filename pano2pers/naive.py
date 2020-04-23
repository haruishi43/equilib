#!/usr/bin/env python3

import numpy as np


def create_coord(
    height: int,
    width: int,
) -> None:
    r"""Create mesh coordinate grid
    """
    _xs = np.linspace(0, width-1, width)
    _ys = np.linspace(0, height-1, height)
    xs, ys = np.meshgrid(_xs, _ys)
    zs = np.ones_like(xs)
    coord = np.stack((xs, ys, zs), axis=2)
    return coord


def create_K(h, w, fov_x):
    f = w / (2 * np.tan(np.radians(fov_x) / 2))
    K = np.array([
        [f, 0, w/2],
        [0, f, h/2],
        [0, 0, 1]])
    return K


def create_rot_mat(rot):
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


def pixel_wise_rot(rot_coord):
    a = np.arctan2(rot_coord[:, :, 0], rot_coord[:, :, 2])
    b = np.arcsin(rot_coord[:, :, 1] / np.linalg.norm(rot_coord, axis=2))
    return a, b


def grid_sample(img, grid):
    r"""
    """
    #TODO: Implement
    return img


if __name__ == "__main__":
    import os
    import os.path as osp

    from PIL import Image

    data_path = osp.join('..', 'data')
    pano_path = osp.join(data_path, 'pano2.png')

    pano_img = Image.open(pano_path)
    pano = np.asarray(pano_img)

    pano = np.transpose(pano, (2,0,1))

    _, h_pano, w_pano = pano.shape
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

    grid = np.stack((ui, uj), axis=2)
    print(grid.shape)