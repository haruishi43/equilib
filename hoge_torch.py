from typing import List

import numpy as np
import torch


def pano2pers_grid(size: List[int] = None, rot: List[float] = None, f: List[float] = None) -> torch.Tensor:
    """
    パノラマ画像をパースペクティブに切り出す
    torch.nn.functional.grid_sample関数用のgridsを出力します

    :param size: [height, width]　切り出したあとの画像サイズ
    :param rot: [yaw, pitch, roll] 弧度法
    :param f: [f_x, f_y]　焦点距離
    :return: torch.Tensor, shape: (1, height, width, 2)
    """
    f = f or [1, 1]
    rot = rot or [0, 0, 0]
    size = size or [224, 224]

    rot_yaw, rot_pitch, rot_roll = rot
    f_x, f_y = f

    v, u = torch.meshgrid([torch.linspace(-1, 1, s) for s in size])
    u = u / f_x
    v = v / f_y
    w = torch.ones_like(u)
    coord = torch.stack((u, v, w), dim=2)

    R_yaw = [
        [np.cos(rot_yaw), 0, -np.sin(rot_yaw)],
        [0, 1, 0],
        [np.sin(rot_yaw), 0, np.cos(rot_yaw)]
    ]
    R_pitch = [
        [1, 0, 0],
        [0, np.cos(rot_pitch), -np.sin(rot_pitch)],
        [0, np.sin(rot_pitch), np.cos(rot_pitch)]
    ]
    R_roll = [
        [np.cos(rot_roll), -np.sin(rot_roll), 0],
        [np.sin(rot_roll), np.cos(rot_roll), 0],
        [0, 0, 1]
    ]
    R = torch.Tensor(R_roll) @ torch.Tensor(R_pitch) @ torch.Tensor(R_yaw)

    roted_coord = coord @ R

    x_sin = roted_coord[:, :, 0] / torch.norm(roted_coord[:, :, [0, 2]], dim=2)
    y_sin = roted_coord[:, :, 1] / torch.norm(roted_coord, dim=2)

    x_raw = torch.asin(x_sin)
    x_raw = torch.where(roted_coord[:, :, 2] < 0, np.pi - x_raw, x_raw)
    y_raw = torch.asin(y_sin)

    x    = x_raw / np.pi
    x = torch.remainder(x, 2) - 1
    y = 2 * y_raw / np.pi

    grid = torch.stack((x, y), 2).unsqueeze(0)
    return grid


def pers2pano_grid(size: List[int] = None, rot: List[float] = None, f: List[float] = None) -> torch.Tensor:
    """
    パースペクティブ画像をパノラマ画像にもどす
    torch.nn.functional.grid_sample関数用のgridsを出力します

    :param size: [height, width]　戻した先のパノラマ画像サイズ
    :param rot: [yaw, pitch, roll] 弧度法
    :param f: [f_x, f_y]　焦点距離
    :return: torch.Tensor, shape: (1, height, width, 2)
    """
    f = f or [1, 1]
    rot = rot or [0, 0, 0]
    size = size or [960, 1920]

    rot_yaw, rot_pitch, rot_roll = rot
    f_x, f_y = f

    phi, theta = torch.meshgrid(
        [torch.linspace(-np.pi / 2, np.pi / 2, size[0] + 2)[1:-1], torch.linspace(0, np.pi * 2, size[1] + 2)[1:-1]])

    u = torch.sin(theta) * torch.cos(phi)
    v = torch.sin(phi)
    w = torch.cos(theta) * torch.cos(phi)
    coord = torch.stack((u, v, w), dim=2)

    R_yaw = [
        [np.cos(rot_yaw), 0, np.sin(rot_yaw)],
        [0, 1, 0],
        [-np.sin(rot_yaw), 0, np.cos(rot_yaw)]
    ]
    R_pitch = [
        [1, 0, 0],
        [0, np.cos(rot_pitch), np.sin(rot_pitch)],
        [0, -np.sin(rot_pitch), np.cos(rot_pitch)]
    ]
    R_roll = [
        [np.cos(rot_roll), np.sin(rot_roll), 0],
        [-np.sin(rot_roll), np.cos(rot_roll), 0],
        [0, 0, 1]
    ]
    R = torch.Tensor(R_yaw) @ torch.Tensor(R_pitch) @ torch.Tensor(R_roll)

    roted_coord = coord @ R

    x = torch.where(roted_coord[:, :, 2] > 0, roted_coord[:, :, 0] / roted_coord[:, :, 2],
                    torch.full(size, np.nan)) * f_x
    y = torch.where(roted_coord[:, :, 2] > 0, roted_coord[:, :, 1] / roted_coord[:, :, 2],
                    torch.full(size, np.nan)) * f_y

    grid = torch.stack((x, y), 2).unsqueeze(0)
    return grid