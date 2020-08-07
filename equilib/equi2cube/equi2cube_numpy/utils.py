#!/usr/bin/env python3

import numpy as np

from scipy.ndimage import map_coordinates


def xyzcube(face_w: int):
    r"""Return the xyz cordinates of the unit cube in [F R B L U D] format.
    """
    out = np.zeros((face_w, face_w * 6, 3), np.float32)
    rng = np.linspace(-0.5, 0.5, num=face_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(rng, -rng), -1)

    # Front face (z = 0.5)
    out[:, 0*face_w:1*face_w, [0, 1]] = grid
    out[:, 0*face_w:1*face_w, 2] = 0.5

    # Right face (x = 0.5)
    out[:, 1*face_w:2*face_w, [2, 1]] = grid
    out[:, 1*face_w:2*face_w, 0] = 0.5

    # Back face (z = -0.5)
    out[:, 2*face_w:3*face_w, [0, 1]] = grid
    out[:, 2*face_w:3*face_w, 2] = -0.5

    # Left face (x = -0.5)
    out[:, 3*face_w:4*face_w, [2, 1]] = grid
    out[:, 3*face_w:4*face_w, 0] = -0.5

    # Up face (y = 0.5)
    out[:, 4*face_w:5*face_w, [0, 2]] = grid
    out[:, 4*face_w:5*face_w, 1] = 0.5

    # Down face (y = -0.5)
    out[:, 5*face_w:6*face_w, [0, 2]] = grid
    out[:, 5*face_w:6*face_w, 1] = -0.5

    return out


def xyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x**2 + z**2)
    v = np.arctan2(y, c)

    return np.concatenate([u, v], axis=-1)


def uv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = np.split(uv, 2, axis=-1)
    coor_x = (u / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (-v / np.pi + 0.5) * h - 0.5

    return np.concatenate([coor_x, coor_y], axis=-1)


def sample_equirec(e_img, coor_xy, order):
    w = e_img.shape[1]
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    pad_u = np.roll(e_img[[0]], w // 2, 1)
    pad_d = np.roll(e_img[[-1]], w // 2, 1)
    e_img = np.concatenate([e_img, pad_d, pad_u], 0)
    return map_coordinates(
        e_img, [coor_y, coor_x],
        order=order, mode='wrap')[..., 0]


def cube_h2list(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    return np.split(cube_h, 6, axis=1)


def cube_h2dict(cube_h):
    cube_list = cube_h2list(cube_h)
    return dict(
        [(k, cube_list[i])
            for i, k in enumerate(['F', 'R', 'B', 'L', 'U', 'D'])]
    )


def cube_h2dice(cube_h):
    assert cube_h.shape[0] * 6 == cube_h.shape[1]
    w = cube_h.shape[0]
    cube_dice = np.zeros((w * 3, w * 4, cube_h.shape[2]), dtype=cube_h.dtype)
    cube_list = cube_h2list(cube_h)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w] = face
    return cube_dice
