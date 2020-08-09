#!/usr/bin/env python3

import numpy as np

from scipy.ndimage import map_coordinates


def create_rotation_matrix(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    r"""Create Rotation Matrix

    params:
        x: x-axis rotation float
        y: y-axis rotation float
        z: z-axis rotation float

    return:
        rotation matrix: numpy.ndarray
    """
    # calculate rotation about the x-axis
    R_x = np.array([
        [1., 0., 0.],
        [0., np.cos(x), -np.sin(x)],
        [0., np.sin(x), np.cos(x)]])
    # calculate rotation about the y-axis
    R_y = np.array([
        [np.cos(y), 0., np.sin(y)],
        [0., 1., 0.],
        [-np.sin(y), 0., np.cos(y)]])
    # calculate rotation about the z-axis
    R_z = np.array([
        [np.cos(z), -np.sin(z), 0.],
        [np.sin(z), np.cos(z), 0.],
        [0., 0., 1.]])

    return R_z @ R_y @ R_x


def sample_cubefaces(cube_faces, tp, coor_y, coor_x, order):
    cube_faces = cube_faces.copy()
    cube_faces[1] = np.flip(cube_faces[1], 1)
    cube_faces[2] = np.flip(cube_faces[2], 1)
    cube_faces[4] = np.flip(cube_faces[4], 0)

    # Pad up down
    pad_ud = np.zeros((6, 2, cube_faces.shape[2]))
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], 1)

    # Pad left right
    pad_lr = np.zeros((6, cube_faces.shape[1], 2))
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], 2)

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode='wrap')


def cube_list2h(cube_list: list):
    assert len(cube_list) == 6
    assert sum(face.shape == cube_list[0].shape for face in cube_list) == 6
    return np.concatenate(cube_list, axis=-1)


def cube_dict2h(cube_dict: dict, face_k=['F', 'R', 'B', 'L', 'U', 'D']):
    assert len(face_k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])


def cube_dice2h(cube_dice: np.ndarray):
    r"""dice to horizion
    params:
    cube_dice: (C, H, W)
    """
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    w = cube_dice.shape[-2] // 3
    assert cube_dice.shape[-2] == w * 3 and cube_dice.shape[-1] == w * 4
    cube_h = np.zeros(
        (cube_dice.shape[2], w, w * 6),
        dtype=cube_dice.dtype
    )
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[:, sy*w:(sy+1)*w, sx*w:(sx+1)*w]
        cube_h[:, :, i*w:(i+1)*w] = face
    return cube_h
